#!/usr/local/bin/python3
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import cv2

import vacode

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval', """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 37600,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")




def recognition():
  with tf.Graph().as_default():
    image_size = vacode.IMAGE_SIZE
    image = tf.placeholder(tf.uint8, [image_size, image_size])
    float_image = tf.cast(image, tf.float32)
    reshape_img = tf.reshape(float_image, [image_size, image_size, 1])

  # Subtract off the mean and divide by the variance of the pixels.
    standardization_image = tf.image.per_image_standardization(reshape_img)
    image_batch = tf.reshape(standardization_image, [1, image_size, image_size,
                                                     1])


    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = vacode.inference(image_batch)

    # Calculate predictions.
    argmax = tf.argmax(logits, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        vacode.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    ##print (variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)


    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    with tf.Session() as sess:
      ##init = tf.global_variables_initializer()
      ##sess.run(init)
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        print (ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print ("loaded")
      else:
        print('No checkpoint file found')
        return

      dir_path = "images"
      for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        print (file_path)
        image_read = cv2.imread(file_path)
        img_flip = cv2.flip(image_read, 0)
        feed_dict = {image: img_flip[:,:,0]}
        prediction = sess.run(argmax, feed_dict = feed_dict )
        print (prediction)



def main(argv=None):  # pylint: disable=unused-argument
  if gfile.Exists(FLAGS.eval_dir):
    gfile.DeleteRecursively(FLAGS.eval_dir)
  gfile.MakeDirs(FLAGS.eval_dir)
  recognition()


if __name__ == '__main__':
  tf.app.run()
