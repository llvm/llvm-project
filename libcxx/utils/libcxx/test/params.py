#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *

_allStandards = ['c++03', 'c++11', 'c++14', 'c++17', 'c++2a']

DEFAULT_PARAMETERS = [
  # Core parameters of the test suite
  Parameter(name='std', choices=_allStandards, type=str,
            help="The version of the standard to compile the test suite with.",
            default=lambda cfg: next(s for s in reversed(_allStandards) if hasCompileFlag(cfg, '-std='+s)),
            actions=lambda std: [
              AddFeature(std),
              AddCompileFlag('-std={}'.format(std)),
            ]),

  Parameter(name='enable_exceptions', choices=[True, False], type=bool, default=True,
            help="Whether to enable exceptions when compiling the test suite.",
            actions=lambda exceptions: [] if exceptions else [
              AddFeature('no-exceptions'),
              AddCompileFlag('-fno-exceptions')
            ]),

  Parameter(name='enable_rtti', choices=[True, False], type=bool, default=True,
            help="Whether to enable RTTI when compiling the test suite.",
            actions=lambda rtti: [] if rtti else [
              AddFeature('no-rtti'),
              AddCompileFlag('-fno-rtti')
            ]),

  Parameter(name='stdlib', choices=['libc++', 'libstdc++', 'msvc'], type=str, default='libc++',
            help="The C++ Standard Library implementation being tested.",
            actions=lambda stdlib: [
              AddFeature(stdlib)
            ]),

  # Parameters to enable or disable parts of the test suite
  Parameter(name='enable_filesystem', choices=[True, False], type=bool, default=True,
            help="Whether to enable tests for the C++ <filesystem> library.",
            actions=lambda filesystem: [] if filesystem else [
              AddFeature('c++filesystem-disabled')
            ]),

  Parameter(name='enable_experimental', choices=[True, False], type=bool, default=False,
            help="Whether to enable tests for experimental C++ libraries (typically Library Fundamentals TSes).",
            actions=lambda experimental: [] if not experimental else [
              AddFeature('c++experimental'),
              AddLinkFlag('-lc++experimental')
            ]),

  Parameter(name='long_tests', choices=[True, False], type=bool, default=True,
            help="Whether to enable tests that take longer to run. This can be useful when running on a very slow device.",
            actions=lambda enabled: [] if not enabled else [
              AddFeature('long_tests')
            ]),

  Parameter(name='enable_debug_tests', choices=[True, False], type=bool, default=True,
            help="Whether to enable tests that exercise the libc++ debugging mode.",
            actions=lambda enabled: [] if enabled else [
              AddFeature('libcxx-no-debug-mode')
            ]),
]
