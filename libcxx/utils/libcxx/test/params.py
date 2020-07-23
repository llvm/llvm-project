#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *

_allStandards = ['c++98', 'c++03', 'c++11', 'c++14', 'c++17', 'c++2a']

DEFAULT_PARAMETERS = [
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

  Parameter(name='enable_rtti', choices=[True, False], type=bool, default=True,
            help="Whether to enable RTTI when compiling the test suite.",
            feature=lambda rtti: None if rtti else
              Feature(name='-fno-rtti', compileFlag='-fno-rtti')),

  Parameter(name='stdlib', choices=['libc++', 'libstdc++', 'msvc'], type=str, default='libc++',
            help="The C++ Standard Library implementation being tested.",
            feature=lambda stdlib: Feature(name=stdlib)),

  # Parameters to enable or disable parts of the test suite
  Parameter(name='enable_filesystem', choices=[True, False], type=bool, default=True,
            help="Whether to enable tests for the C++ <filesystem> library.",
            feature=lambda filesystem: None if filesystem else
              Feature(name='c++filesystem-disabled')),

  Parameter(name='enable_experimental', choices=[True, False], type=bool, default=False,
          help="Whether to enable tests for experimental C++ libraries (typically Library Fundamentals TSes).",
          feature=lambda experimental: None if not experimental else
            Feature(name='c++experimental', linkFlag='-lc++experimental')),

  Parameter(name='long_tests', choices=[True, False], type=bool, default=True,
            help="Whether to tests that take longer to run. This can be useful when running on a very slow device.",
            feature=lambda enabled: Feature(name='long_tests') if enabled else None),
]
