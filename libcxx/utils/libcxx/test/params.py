#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *
from libcxx.test.features import _isMSVC
import re

_warningFlags = [
  '-Werror',
  '-Wall',
  '-Wextra',
  '-Wshadow',
  '-Wundef',
  '-Wno-unused-command-line-argument',
  '-Wno-attributes',
  '-Wno-pessimizing-move',
  '-Wno-c++11-extensions',
  '-Wno-noexcept-type',
  '-Wno-aligned-allocation-unavailable',
  '-Wno-atomic-alignment',

  # GCC warns about places where we might want to add sized allocation/deallocation
  # functions, but we know better what we're doing/testing in the test suite.
  '-Wno-sized-deallocation',

  # Turn off warnings about user-defined literals with reserved suffixes. Those are
  # just noise since we are testing the Standard Library itself.
  '-Wno-literal-suffix', # GCC
  '-Wno-user-defined-literals', # Clang

  # GCC warns about this when TEST_IS_CONSTANT_EVALUATED is used on a non-constexpr
  # function. (This mostely happens in C++11 mode.)
  # TODO(mordante) investigate a solution for this issue.
  '-Wno-tautological-compare',

  # -Wstringop-overread seems to be a bit buggy currently
  '-Wno-stringop-overread',

  # These warnings should be enabled in order to support the MSVC
  # team using the test suite; They enable the warnings below and
  # expect the test suite to be clean.
  '-Wsign-compare',
  '-Wunused-variable',
  '-Wunused-parameter',
  '-Wunreachable-code',
  '-Wno-unused-local-typedef',
]

_allStandards = ['c++03', 'c++11', 'c++14', 'c++17', 'c++20', 'c++2b']
def getStdFlag(cfg, std):
  fallbacks = {
    'c++11': 'c++0x',
    'c++14': 'c++1y',
    'c++17': 'c++1z',
    'c++20': 'c++2a',
  }
  if hasCompileFlag(cfg, '-std='+std):
    return '-std='+std
  if std in fallbacks and hasCompileFlag(cfg, '-std='+fallbacks[std]):
    return '-std='+fallbacks[std]
  return None

DEFAULT_PARAMETERS = [
  Parameter(name='target_triple', type=str,
            help="The target triple to compile the test suite for. This must be "
                 "compatible with the target that the tests will be run on.",
            actions=lambda triple: filter(None, [
              AddFeature('target={}'.format(triple)),
              AddFlagIfSupported('--target={}'.format(triple)),
              AddSubstitution('%{triple}', triple)
            ])),

  Parameter(name='std', choices=_allStandards, type=str,
            help="The version of the standard to compile the test suite with.",
            default=lambda cfg: next(s for s in reversed(_allStandards) if getStdFlag(cfg, s)),
            actions=lambda std: [
              AddFeature(std),
              AddCompileFlag(lambda cfg: getStdFlag(cfg, std)),
            ]),

  Parameter(name='enable_modules', choices=[True, False], type=bool, default=False,
            help="Whether to build the test suite with Clang modules enabled.",
            actions=lambda modules: [
              AddFeature('modules-build'),
              AddCompileFlag('-fmodules'),
              AddCompileFlag('-fcxx-modules'), # AppleClang disregards -fmodules entirely when compiling C++. This enables modules for C++.
            ] if modules else []),

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

  Parameter(name='stdlib', choices=['llvm-libc++', 'apple-libc++', 'libstdc++', 'msvc'], type=str, default='llvm-libc++',
            help="""The C++ Standard Library implementation being tested.

                 Note that this parameter can also be used to encode different 'flavors' of the same
                 standard library, such as libc++ as shipped by a different vendor, if it has different
                 properties worth testing.

                 The Standard libraries currently supported are:
                 - llvm-libc++: The 'upstream' libc++ as shipped with LLVM.
                 - apple-libc++: libc++ as shipped by Apple. This is basically like the LLVM one, but
                                 there are a few differences like installation paths and the use of
                                 universal dylibs.
                 - libstdc++: The GNU C++ library typically shipped with GCC.
                 - msvc: The Microsoft implementation of the C++ Standard Library.
                """,
            actions=lambda stdlib: filter(None, [
              AddFeature('stdlib={}'.format(stdlib)),
              # Also add an umbrella feature 'stdlib=libc++' for all flavors of libc++, to simplify
              # the test suite.
              AddFeature('stdlib=libc++') if re.match('.+-libc\+\+', stdlib) else None
            ])),

  Parameter(name='enable_warnings', choices=[True, False], type=bool, default=True,
            help="Whether to enable warnings when compiling the test suite.",
            actions=lambda warnings: [] if not warnings else
              [AddOptionalWarningFlag(w) for w in _warningFlags] +
              [AddCompileFlag('-D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER')]
            ),

  Parameter(name='use_sanitizer', choices=['', 'Address', 'Undefined', 'Memory', 'MemoryWithOrigins', 'Thread', 'DataFlow', 'Leaks'], type=str, default='',
            help="An optional sanitizer to enable when building and running the test suite.",
            actions=lambda sanitizer: filter(None, [
              AddFlag('-g -fno-omit-frame-pointer') if sanitizer else None,

              AddFlag('-fsanitize=undefined -fno-sanitize=float-divide-by-zero -fno-sanitize-recover=all') if sanitizer == 'Undefined' else None,
              AddFeature('ubsan')                                                                          if sanitizer == 'Undefined' else None,

              AddFlag('-fsanitize=address') if sanitizer == 'Address' else None,
              AddFeature('asan')            if sanitizer == 'Address' else None,

              AddFlag('-fsanitize=memory')               if sanitizer in ['Memory', 'MemoryWithOrigins'] else None,
              AddFeature('msan')                         if sanitizer in ['Memory', 'MemoryWithOrigins'] else None,
              AddFlag('-fsanitize-memory-track-origins') if sanitizer == 'MemoryWithOrigins' else None,

              AddFlag('-fsanitize=thread') if sanitizer == 'Thread' else None,
              AddFeature('tsan')           if sanitizer == 'Thread' else None,

              AddFlag('-fsanitize=dataflow') if sanitizer == 'DataFlow' else None,
              AddFlag('-fsanitize=leaks') if sanitizer == 'Leaks' else None,

              AddFeature('sanitizer-new-delete') if sanitizer in ['Address', 'Memory', 'MemoryWithOrigins', 'Thread'] else None,
            ])),

  Parameter(name='enable_experimental', choices=[True, False], type=bool, default=True,
            help="Whether to enable tests for experimental C++ Library features.",
            actions=lambda experimental: [] if not experimental else [
              # When linking in MSVC mode via the Clang driver, a -l<foo>
              # maps to <foo>.lib, so we need to use -llibc++experimental here
              # to make it link against the static libc++experimental.lib.
              # We can't check for the feature 'msvc' in available_features
              # as those features are added after processing parameters.
              #
              # TODO: Switch to using the appropriate experimental compiler flag once
              #       all compilers we support implement that flag.
              AddFeature('c++experimental'),
              PrependLinkFlag(lambda cfg: '-llibc++experimental' if _isMSVC(cfg) else '-lc++experimental'),
              AddCompileFlag('-D_LIBCPP_ENABLE_EXPERIMENTAL'),
            ]),

  Parameter(name='long_tests', choices=[True, False], type=bool, default=True,
            help="Whether to enable tests that take longer to run. This can be useful when running on a very slow device.",
            actions=lambda enabled: [] if not enabled else [
              AddFeature('long_tests')
            ]),

  Parameter(name='enable_assertions', choices=[True, False], type=bool, default=False,
            help="Whether to enable assertions when compiling the test suite. This is only meaningful when "
                 "running the tests against libc++.",
            actions=lambda assertions: [
              AddCompileFlag('-D_LIBCPP_ENABLE_ASSERTIONS=1'),
              AddFeature('libcpp-has-assertions')
            ] if assertions else []),

  Parameter(name='additional_features', type=list, default=[],
            help="A comma-delimited list of additional features that will be enabled when running the tests. "
                 "This should be used sparingly since specifying ad-hoc features manually is error-prone and "
                 "brittle in the long run as changes are made to the test suite.",
            actions=lambda features: [AddFeature(f) for f in features]),

  Parameter(name='enable_transitive_includes', choices=[True, False], type=bool, default=True,
            help="Whether to enable backwards-compatibility transitive includes when running the tests. This "
                 "is provided to ensure that the trimmed-down version of libc++ does not bit-rot in between "
                 "points at which we bulk-remove transitive includes.",
            actions=lambda enabled: [] if enabled else [
              AddFeature('transitive-includes-disabled'),
              AddCompileFlag('-D_LIBCPP_REMOVE_TRANSITIVE_INCLUDES')
            ]),
]

DEFAULT_PARAMETERS += [
  Parameter(name='use_system_cxx_lib', choices=[True, False], type=bool, default=False,
            help="""
    Whether the test suite is being *run* against the library shipped on the
    target triple in use, as opposed to the trunk library.

    When vendor-specific availability annotations are enabled, we add the
    'use_system_cxx_lib' Lit feature to allow writing XFAIL or UNSUPPORTED
    markup for tests that are known to fail on a particular triple.

    That feature can be used to XFAIL a test that fails when deployed on (or is
    compiled for) an older system. For example, if the test exhibits a bug in the
    libc on a particular system version, or if the test uses a symbol that is not
    available on an older version of the dylib, it can be marked as XFAIL with
    the above feature.

    It is sometimes useful to check that a test fails specifically when compiled
    for a given deployment target. For example, this is the case when testing
    availability markup, where we want to make sure that using the annotated
    facility on a deployment target that doesn't support it will fail at compile
    time, not at runtime. This can be achieved by creating a `.compile.pass.cpp`
    and XFAILing it for the right deployment target. If the test doesn't fail at
    compile-time like it's supposed to, the test will XPASS. Another option is to
    create a `.verify.cpp` test that checks for the right errors, and mark that
    test as requiring `use_system_cxx_lib && <target>`.
    """,
    actions=lambda useSystem: [
      AddFeature('use_system_cxx_lib')
    ] if useSystem else [
      # If we're testing upstream libc++, disable availability markup,
      # which is not relevant for non-shipped flavors of libc++.
      AddCompileFlag('-D_LIBCPP_DISABLE_AVAILABILITY')
    ])
]
