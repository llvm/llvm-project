#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *

parameters = [
  Parameter(name='enable_exceptions', choices=[True, False], type=bool, default=True,
            help="Whether to enable exceptions when compiling the test suite.",
            feature=lambda exceptions: None if exceptions else
              Feature(name='no-exceptions', compileFlag='-fno-exceptions')),

  # Parameters to enable or disable parts of the test suite
  Parameter(name='enable_filesystem', choices=[True, False], type=bool, default=True,
            help="Whether to enable tests for the C++ <filesystem> library.",
            feature=lambda filesystem: None if filesystem else
              Feature(name='c++filesystem-disabled')),
]
