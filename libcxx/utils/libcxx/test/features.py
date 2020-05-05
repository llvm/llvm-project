#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *
import sys

_isClang      = lambda cfg: '__clang__' in compilerMacros(cfg) and '__apple_build_version__' not in compilerMacros(cfg)
_isAppleClang = lambda cfg: '__apple_build_version__' in compilerMacros(cfg)
_isGCC        = lambda cfg: '__GNUC__' in compilerMacros(cfg) and '__clang__' not in compilerMacros(cfg)

features = [
  Feature(name='fcoroutines-ts', compileFlag='-fcoroutines-ts',
          when=lambda cfg: hasCompileFlag(cfg, '-fcoroutines-ts') and
                           featureTestMacros(cfg, flags='-fcoroutines-ts').get('__cpp_coroutines', 0) >= 201703),

  Feature(name='thread-safety',                 when=lambda cfg: hasCompileFlag(cfg, '-Werror=thread-safety'), compileFlag='-Werror=thread-safety'),
  Feature(name='has-fblocks',                   when=lambda cfg: hasCompileFlag(cfg, '-fblocks')),
  Feature(name='-fsized-deallocation',          when=lambda cfg: hasCompileFlag(cfg, '-fsized-deallocation')),
  Feature(name='-faligned-allocation',          when=lambda cfg: hasCompileFlag(cfg, '-faligned-allocation')),
  Feature(name='fdelayed-template-parsing',     when=lambda cfg: hasCompileFlag(cfg, '-fdelayed-template-parsing')),
  Feature(name='libcpp-no-if-constexpr',        when=lambda cfg: '__cpp_if_constexpr' not in featureTestMacros(cfg)),
  Feature(name='libcpp-no-structured-bindings', when=lambda cfg: '__cpp_structured_bindings' not in featureTestMacros(cfg)),
  Feature(name='libcpp-no-deduction-guides',    when=lambda cfg: featureTestMacros(cfg).get('__cpp_deduction_guides', 0) < 201611),
  Feature(name='libcpp-no-concepts',            when=lambda cfg: featureTestMacros(cfg).get('__cpp_concepts', 0) < 201811),
  Feature(name='has-fobjc-arc',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc') and
                                                                 sys.platform.lower().strip() == 'darwin'), # TODO: this doesn't handle cross-compiling to Apple platforms.
  Feature(name='objective-c++',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc')),
  Feature(name='diagnose-if-support',           when=lambda cfg: hasCompileFlag(cfg, '-Wuser-defined-warnings'), compileFlag='-Wuser-defined-warnings'),
  Feature(name='modules-support',               when=lambda cfg: hasCompileFlag(cfg, '-fmodules')),

  Feature(name='apple-clang',                                                                                                      when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}'.format(**compilerMacros(cfg)),                                          when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}.{__clang_minor__}'.format(**compilerMacros(cfg)),                        when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}'.format(**compilerMacros(cfg)), when=_isAppleClang),

  Feature(name='clang',                                                                                                            when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}'.format(**compilerMacros(cfg)),                                                when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}.{__clang_minor__}'.format(**compilerMacros(cfg)),                              when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}'.format(**compilerMacros(cfg)),       when=_isClang),

  Feature(name='gcc',                                                                                                              when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}'.format(**compilerMacros(cfg)),                                                         when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}.{__GNUC_MINOR__}'.format(**compilerMacros(cfg)),                                        when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}.{__GNUC_MINOR__}.{__GNUC_PATCHLEVEL__}'.format(**compilerMacros(cfg)),                  when=_isGCC),
]

# Deduce and add the test features that that are implied by the #defines in
# the <__config_site> header.
#
# For each macro of the form `_LIBCPP_XXX_YYY_ZZZ` defined below that
# is defined after including <__config_site>, add a Lit feature called
# `libcpp-xxx-yyy-zzz`. When a macro is defined to a specific value
# (e.g. `_LIBCPP_ABI_VERSION=2`), the feature is `libcpp-xxx-yyy-zzz=<value>`.
macros = [
  '_LIBCPP_HAS_NO_GLOBAL_FILESYSTEM_NAMESPACE',
  '_LIBCPP_HAS_NO_MONOTONIC_CLOCK',
  '_LIBCPP_HAS_NO_STDIN',
  '_LIBCPP_HAS_NO_STDOUT',
  '_LIBCPP_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS',
  '_LIBCPP_HAS_NO_THREADS',
  '_LIBCPP_HAS_THREAD_API_EXTERNAL',
  '_LIBCPP_HAS_THREAD_API_PTHREAD',
  '_LIBCPP_NO_VCRUNTIME',
  '_LIBCPP_ABI_VERSION',
  '_LIBCPP_ABI_UNSTABLE'
]
for macro in macros:
  features += [
    Feature(name=lambda cfg, macro=macro: macro.lower()[1:].replace('_', '-') + (
              '={}'.format(compilerMacros(cfg)[macro]) if compilerMacros(cfg)[macro] else ''
            ),
            when=lambda cfg, macro=macro: macro in compilerMacros(cfg),

            # FIXME: This is a hack that should be fixed using module maps.
            # If modules are enabled then we have to lift all of the definitions
            # in <__config_site> onto the command line.
            compileFlag=lambda cfg, macro=macro: '-Wno-macro-redefined -D{}'.format(macro) + (
              '={}'.format(compilerMacros(cfg)[macro]) if compilerMacros(cfg)[macro] else ''
            )
    )
  ]
