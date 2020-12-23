#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *
import re
import shutil
import sys

_isClang      = lambda cfg: '__clang__' in compilerMacros(cfg) and '__apple_build_version__' not in compilerMacros(cfg)
_isAppleClang = lambda cfg: '__apple_build_version__' in compilerMacros(cfg)
_isGCC        = lambda cfg: '__GNUC__' in compilerMacros(cfg) and '__clang__' not in compilerMacros(cfg)

DEFAULT_FEATURES = [
  Feature(name='fcoroutines-ts',
          when=lambda cfg: hasCompileFlag(cfg, '-fcoroutines-ts') and
                           featureTestMacros(cfg, flags='-fcoroutines-ts').get('__cpp_coroutines', 0) >= 201703,
          actions=[AddCompileFlag('-fcoroutines-ts')]),

  Feature(name='thread-safety',
          when=lambda cfg: hasCompileFlag(cfg, '-Werror=thread-safety'),
          actions=[AddCompileFlag('-Werror=thread-safety')]),

  Feature(name='diagnose-if-support',
          when=lambda cfg: hasCompileFlag(cfg, '-Wuser-defined-warnings'),
          actions=[AddCompileFlag('-Wuser-defined-warnings')]),

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

  # Note: We use a custom modules cache path to make sure that we don't reuse
  #       the default one, which can be shared across builds. This is important
  #       because we define macros in headers files, and a change in these macros
  #       doesn't seem to invalidate modules cache entries, which means we could
  #       build against now-invalid cached headers from a previous build.
  Feature(name='modules-support',
          when=lambda cfg: hasCompileFlag(cfg, '-fmodules'),
          actions=lambda cfg: [AddCompileFlag('-fmodules-cache-path=%t/ModuleCache')]),

  Feature(name='non-lockfree-atomics',
          when=lambda cfg: sourceBuilds(cfg, """
            #include <atomic>
            struct Large { int storage[100]; };
            std::atomic<Large> x;
            int main(int, char**) { (void)x.load(); return 0; }
          """)),
  # TODO: Remove this feature once compiler-rt includes __atomic_is_lockfree()
  # on all supported platforms.
  Feature(name='is-lockfree-runtime-function',
          when=lambda cfg: sourceBuilds(cfg, """
            #include <atomic>
            struct Large { int storage[100]; };
            std::atomic<Large> x;
            int main(int, char**) { return x.is_lock_free(); }
          """)),

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
macros = {
  '_LIBCPP_HAS_NO_GLOBAL_FILESYSTEM_NAMESPACE': 'libcpp-has-no-global-filesystem-namespace',
  '_LIBCPP_HAS_NO_MONOTONIC_CLOCK': 'libcpp-has-no-monotonic-clock',
  '_LIBCPP_HAS_NO_STDIN': 'libcpp-has-no-stdin',
  '_LIBCPP_HAS_NO_STDOUT': 'libcpp-has-no-stdout',
  '_LIBCPP_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS': 'libcpp-has-no-thread-unsafe-c-functions',
  '_LIBCPP_HAS_NO_THREADS': 'libcpp-has-no-threads',
  '_LIBCPP_HAS_THREAD_API_EXTERNAL': 'libcpp-has-thread-api-external',
  '_LIBCPP_HAS_THREAD_API_PTHREAD': 'libcpp-has-thread-api-pthread',
  '_LIBCPP_NO_VCRUNTIME': 'libcpp-no-vcruntime',
  '_LIBCPP_ABI_VERSION': 'libcpp-abi-version',
  '_LIBCPP_ABI_UNSTABLE': 'libcpp-abi-unstable',
  '_LIBCPP_HAS_NO_RANDOM_DEVICE': 'libcpp-has-no-random-device',
  '_LIBCPP_HAS_NO_LOCALIZATION': 'libcpp-has-no-localization',
}
for macro, feature in macros.items():
  DEFAULT_FEATURES += [
    Feature(name=lambda cfg, m=macro, f=feature: f + (
              '={}'.format(compilerMacros(cfg)[m]) if compilerMacros(cfg)[m] else ''
            ),
            when=lambda cfg, m=macro: m in compilerMacros(cfg),

            # FIXME: This is a hack that should be fixed using module maps.
            # If modules are enabled then we have to lift all of the definitions
            # in <__config_site> onto the command line.
            actions=lambda cfg, m=macro: [
              AddCompileFlag('-Wno-macro-redefined -D{}'.format(m) + (
                '={}'.format(compilerMacros(cfg)[m]) if compilerMacros(cfg)[m] else ''
              ))
            ]
    )
  ]


# Mapping from canonical locale names (used in the tests) to possible locale
# names on various systems. Each locale is considered supported if any of the
# alternative names is supported.
locales = {
  'en_US.UTF-8':     ['en_US.UTF-8', 'en_US.utf8', 'English_United States.1252'],
  'fr_FR.UTF-8':     ['fr_FR.UTF-8', 'fr_FR.utf8', 'French_France.1252'],
  'ru_RU.UTF-8':     ['ru_RU.UTF-8', 'ru_RU.utf8', 'Russian_Russia.1251'],
  'zh_CN.UTF-8':     ['zh_CN.UTF-8', 'zh_CN.utf8', 'Chinese_China.936'],
  'fr_CA.ISO8859-1': ['fr_CA.ISO8859-1', 'French_Canada.1252'],
  'cs_CZ.ISO8859-2': ['cs_CZ.ISO8859-2', 'Czech_Czech Republic.1250']
}
for locale, alts in locales.items():
  # Note: Using alts directly in the lambda body here will bind it to the value at the
  # end of the loop. Assigning it to a default argument works around this issue.
  DEFAULT_FEATURES.append(Feature(name='locale.{}'.format(locale),
                                  when=lambda cfg, alts=alts: hasAnyLocale(cfg, alts)))


# Add features representing the platform name: darwin, linux, windows, etc...
DEFAULT_FEATURES += [
  Feature(name='darwin', when=lambda cfg: '__APPLE__' in compilerMacros(cfg)),
  Feature(name='windows', when=lambda cfg: '_WIN32' in compilerMacros(cfg)),
  Feature(name='linux', when=lambda cfg: '__linux__' in compilerMacros(cfg)),
  Feature(name='netbsd', when=lambda cfg: '__NetBSD__' in compilerMacros(cfg))
]


# Detect whether GDB is on the system, and if so add a substitution to access it.
DEFAULT_FEATURES += [
  Feature(name='host-has-gdb',
    when=lambda cfg: shutil.which('gdb') is not None,
    actions=[AddSubstitution('%{gdb}', lambda cfg: shutil.which('gdb'))]
  )
]


# When vendor-specific availability annotations are enabled, add Lit features
# with various forms of the target triple to make it easier to write XFAIL or
# UNSUPPORTED markup for tests that are known to fail on a particular triple.
#
# More specifically, when the `use_system_cxx_lib` parameter is enabled, then
# assuming the `target_triple` is set to `x86_64-apple-macosx10.12`, the
# following features will be made available:
#   - with_system_cxx_lib=macosx
#   - with_system_cxx_lib=macosx10.12
#   - with_system_cxx_lib=x86_64-apple-macosx10.12
#
# These features can be used to XFAIL a test that fails when deployed on (or is
# compiled for) an older system. For example, if the test exhibits a bug in the
# libc on a particular system version, or if the test uses a symbol that is not
# available on an older version of the dylib, it can be marked as XFAIL with
# one of the above features.
#
# It is sometimes useful to check that a test fails specifically when compiled
# for a given deployment target. For example, this is the case when testing
# availability markup, where we want to make sure that using the annotated
# facility on a deployment target that doesn't support it will fail at compile
# time, not at runtime. This can be achieved by creating a `.compile.pass.cpp`
# and XFAILing it for the right deployment target. If the test doesn't fail at
# compile-time like it's supposed to, the test will XPASS. Another option is to
# create a `.verify.cpp` test that checks for the right errors, and mark that
# test as requiring `with_system_cxx_lib=<something>`.
#
# TODO: This is very unclean -- we assume that the 'use_system_cxx_lib' parameter
#       is set before this feature gets detected, and we also set a dummy name
#       for the main feature. We also take for granted that `target_triple`
#       exists in the config object. This should be refactored so that the
#       'use_system_cxx_lib' Parameter can set these features itself.
def _addSystemCxxLibDeclinations(cfg):
  (arch, vendor, platform) = cfg.target_triple.split('-', 2)
  (sysname, version) = re.match(r'([^0-9]+)([0-9\.]*)', platform).groups()
  return [
    AddFeature('with_system_cxx_lib={}-{}-{}{}'.format(arch, vendor, sysname, version)),
    AddFeature('with_system_cxx_lib={}{}'.format(sysname, version)),
    AddFeature('with_system_cxx_lib={}'.format(sysname)),
  ]
DEFAULT_FEATURES += [
  Feature(name='__dummy_use_system_cxx_lib',
          when=lambda cfg: 'use_system_cxx_lib' in cfg.available_features,
          actions=_addSystemCxxLibDeclinations)
]
