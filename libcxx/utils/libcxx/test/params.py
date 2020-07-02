#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *

_allStandards = ['c++98', 'c++03', 'c++11', 'c++14', 'c++17', 'c++2a']

parameters = [
  # Core parameters of the test suite
  Parameter(name='std', choices=_allStandards, type=str,
            help="The version of the standard to compile the test suite with.",
            default=lambda cfg: next(s for s in reversed(_allStandards) if hasCompileFlag(cfg, '-std='+s)),
            feature=lambda std:
              Feature(name=std, compileFlag='-std={}'.format(std),
                      when=lambda cfg: hasCompileFlag(cfg, '-std={}'.format(std)))),

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
