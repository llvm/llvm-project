#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

from libcxx.test.dsl import *
from lit.BooleanExpression import BooleanExpression
import re
import shutil
import subprocess
import sys

_isClang      = lambda cfg: '__clang__' in compilerMacros(cfg) and '__apple_build_version__' not in compilerMacros(cfg)
_isAppleClang = lambda cfg: '__apple_build_version__' in compilerMacros(cfg)
_isGCC        = lambda cfg: '__GNUC__' in compilerMacros(cfg) and '__clang__' not in compilerMacros(cfg)
_isMSVC       = lambda cfg: '_MSC_VER' in compilerMacros(cfg)
_msvcVersion  = lambda cfg: (int(compilerMacros(cfg)['_MSC_VER']) // 100, int(compilerMacros(cfg)['_MSC_VER']) % 100)

def _getSuitableClangTidy(cfg):
  try:
    # If we didn't build the libcxx-tidy plugin via CMake, we can't run the clang-tidy tests.
    if runScriptExitCode(cfg, ['stat %{test-tools}/clang_tidy_checks/libcxx-tidy.plugin']) != 0:
      return None

    # TODO This should be the last stable release.
    if runScriptExitCode(cfg, ['clang-tidy-16 --version']) == 0:
      return 'clang-tidy-16'

    if int(re.search('[0-9]+', commandOutput(cfg, ['clang-tidy --version'])).group()) >= 16:
      return 'clang-tidy'

  except ConfigurationRuntimeError:
    return None

DEFAULT_FEATURES = [
  Feature(name='thread-safety',
          when=lambda cfg: hasCompileFlag(cfg, '-Werror=thread-safety'),
          actions=[AddCompileFlag('-Werror=thread-safety')]),

  Feature(name='diagnose-if-support',
          when=lambda cfg: hasCompileFlag(cfg, '-Wuser-defined-warnings'),
          actions=[AddCompileFlag('-Wuser-defined-warnings')]),

  # Tests to validate whether the compiler has a way to set the maximum number
  # of steps during constant evaluation. Since the flag differs per compiler
  # store the "valid" flag as a feature. This allows passing the proper compile
  # flag to the compiler:
  # // ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12345678
  # // ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=12345678
  Feature(name='has-fconstexpr-steps',
          when=lambda cfg: hasCompileFlag(cfg, '-fconstexpr-steps=1')),

  Feature(name='has-fconstexpr-ops-limit',
          when=lambda cfg: hasCompileFlag(cfg, '-fconstexpr-ops-limit=1')),

  Feature(name='has-fblocks',                   when=lambda cfg: hasCompileFlag(cfg, '-fblocks')),
  Feature(name='-fsized-deallocation',          when=lambda cfg: hasCompileFlag(cfg, '-fsized-deallocation')),
  Feature(name='-faligned-allocation',          when=lambda cfg: hasCompileFlag(cfg, '-faligned-allocation')),
  Feature(name='fdelayed-template-parsing',     when=lambda cfg: hasCompileFlag(cfg, '-fdelayed-template-parsing')),
  Feature(name='libcpp-no-coroutines',          when=lambda cfg: featureTestMacros(cfg).get('__cpp_impl_coroutine', 0) < 201902),
  Feature(name='has-fobjc-arc',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc') and
                                                                 sys.platform.lower().strip() == 'darwin'), # TODO: this doesn't handle cross-compiling to Apple platforms.
  Feature(name='objective-c++',                 when=lambda cfg: hasCompileFlag(cfg, '-xobjective-c++ -fobjc-arc')),
  Feature(name='verify-support',                when=lambda cfg: hasCompileFlag(cfg, '-Xclang -verify-ignore-unexpected')),

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

  # Some tests rely on creating shared libraries which link in the C++ Standard Library. In some
  # cases, this doesn't work (e.g. if the library was built as a static archive and wasn't compiled
  # as position independent). This feature informs the test suite of whether it's possible to create
  # a shared library in a shell test by using the '-shared' compiler flag.
  #
  # Note: To implement this check properly, we need to make sure that we use something inside the
  # compiled library, not only in the headers. It should be safe to assume that all implementations
  # define `operator new` in the compiled library.
  Feature(name='cant-build-shared-library',
          when=lambda cfg: not sourceBuilds(cfg, """
            void f() { new int(3); }
          """, ['-shared'])),

  # Check for a Windows UCRT bug (fixed in UCRT/Windows 10.0.20348.0):
  # https://developercommunity.visualstudio.com/t/utf-8-locales-break-ctype-functions-for-wchar-type/1653678
  Feature(name='win32-broken-utf8-wchar-ctype',
          when=lambda cfg: not '_LIBCPP_HAS_NO_LOCALIZATION' in compilerMacros(cfg) and '_WIN32' in compilerMacros(cfg) and not programSucceeds(cfg, """
            #include <locale.h>
            #include <wctype.h>
            int main(int, char**) {
              setlocale(LC_ALL, "en_US.UTF-8");
              return towlower(L'\\xDA') != L'\\xFA';
            }
          """)),

  # Check for a Windows UCRT bug (fixed in UCRT/Windows 10.0.19041.0).
  # https://developercommunity.visualstudio.com/t/printf-formatting-with-g-outputs-too/1660837
  Feature(name='win32-broken-printf-g-precision',
          when=lambda cfg: '_WIN32' in compilerMacros(cfg) and not programSucceeds(cfg, """
            #include <stdio.h>
            #include <string.h>
            int main(int, char**) {
              char buf[100];
              snprintf(buf, sizeof(buf), "%#.*g", 0, 0.0);
              return strcmp(buf, "0.");
            }
          """)),

  # Check for Glibc < 2.27, where the ru_RU.UTF-8 locale had
  # mon_decimal_point == ".", which our tests don't handle.
  Feature(name='glibc-old-ru_RU-decimal-point',
          when=lambda cfg: not '_LIBCPP_HAS_NO_LOCALIZATION' in compilerMacros(cfg) and not programSucceeds(cfg, """
            #include <locale.h>
            #include <string.h>
            int main(int, char**) {
              setlocale(LC_ALL, "ru_RU.UTF-8");
              return strcmp(localeconv()->mon_decimal_point, ",");
            }
          """)),

  Feature(name='has-unix-headers',
          when=lambda cfg: sourceBuilds(cfg, """
            #include <unistd.h>
            #include <sys/wait.h>
            int main(int, char**) {
              return 0;
            }
          """)),

  # Whether Bash can run on the executor.
  # This is not always the case, for example when running on embedded systems.
  #
  # For the corner case of bash existing, but it being missing in the path
  # set in %{exec} as "--env PATH=one-single-dir", the executor does find
  # and executes bash, but bash then can't find any other common shell
  # utilities. Test executing "bash -c 'bash --version'" to see if bash
  # manages to find binaries to execute.
  Feature(name='executor-has-no-bash',
          when=lambda cfg: runScriptExitCode(cfg, ['%{exec} bash -c \'bash --version\'']) != 0),
  Feature(name='has-clang-tidy',
          when=lambda cfg: _getSuitableClangTidy(cfg) is not None,
          actions=[AddSubstitution('%{clang-tidy}', lambda cfg: _getSuitableClangTidy(cfg))]),

  Feature(name='apple-clang',                                                                                                      when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}'.format(**compilerMacros(cfg)),                                          when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}.{__clang_minor__}'.format(**compilerMacros(cfg)),                        when=_isAppleClang),
  Feature(name=lambda cfg: 'apple-clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}'.format(**compilerMacros(cfg)), when=_isAppleClang),

  Feature(name='clang',                                                                                                            when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}'.format(**compilerMacros(cfg)),                                                when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}.{__clang_minor__}'.format(**compilerMacros(cfg)),                              when=_isClang),
  Feature(name=lambda cfg: 'clang-{__clang_major__}.{__clang_minor__}.{__clang_patchlevel__}'.format(**compilerMacros(cfg)),       when=_isClang),

  # Note: Due to a GCC bug (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104760), we must disable deprecation warnings
  #       on GCC or spurious diagnostics are issued.
  #
  # TODO:
  # - Enable -Wplacement-new with GCC.
  # - Enable -Wclass-memaccess with GCC.
  Feature(name='gcc',                                                                                                              when=_isGCC,
          actions=[AddCompileFlag('-D_LIBCPP_DISABLE_DEPRECATION_WARNINGS'),
                   AddCompileFlag('-Wno-placement-new'),
                   AddCompileFlag('-Wno-class-memaccess')]),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}'.format(**compilerMacros(cfg)),                                                         when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}.{__GNUC_MINOR__}'.format(**compilerMacros(cfg)),                                        when=_isGCC),
  Feature(name=lambda cfg: 'gcc-{__GNUC__}.{__GNUC_MINOR__}.{__GNUC_PATCHLEVEL__}'.format(**compilerMacros(cfg)),                  when=_isGCC),

  Feature(name='msvc',                                                                                                             when=_isMSVC),
  Feature(name=lambda cfg: 'msvc-{}'.format(*_msvcVersion(cfg)),                                                                   when=_isMSVC),
  Feature(name=lambda cfg: 'msvc-{}.{}'.format(*_msvcVersion(cfg)),                                                                when=_isMSVC),
]

# Deduce and add the test features that that are implied by the #defines in
# the <__config_site> header.
#
# For each macro of the form `_LIBCPP_XXX_YYY_ZZZ` defined below that
# is defined after including <__config_site>, add a Lit feature called
# `libcpp-xxx-yyy-zzz`. When a macro is defined to a specific value
# (e.g. `_LIBCPP_ABI_VERSION=2`), the feature is `libcpp-xxx-yyy-zzz=<value>`.
#
# Note that features that are more strongly tied to libc++ are named libcpp-foo,
# while features that are more general in nature are not prefixed with 'libcpp-'.
macros = {
  '_LIBCPP_HAS_NO_MONOTONIC_CLOCK': 'no-monotonic-clock',
  '_LIBCPP_HAS_NO_THREADS': 'no-threads',
  '_LIBCPP_HAS_THREAD_API_EXTERNAL': 'libcpp-has-thread-api-external',
  '_LIBCPP_HAS_THREAD_API_PTHREAD': 'libcpp-has-thread-api-pthread',
  '_LIBCPP_NO_VCRUNTIME': 'libcpp-no-vcruntime',
  '_LIBCPP_ABI_VERSION': 'libcpp-abi-version',
  '_LIBCPP_HAS_NO_FILESYSTEM_LIBRARY': 'no-filesystem',
  '_LIBCPP_HAS_NO_RANDOM_DEVICE': 'no-random-device',
  '_LIBCPP_HAS_NO_LOCALIZATION': 'no-localization',
  '_LIBCPP_HAS_NO_FSTREAM': 'no-fstream',
  '_LIBCPP_HAS_NO_WIDE_CHARACTERS': 'no-wide-characters',
  '_LIBCPP_HAS_NO_UNICODE': 'libcpp-has-no-unicode',
  '_LIBCPP_ENABLE_DEBUG_MODE': 'libcpp-has-debug-mode',
}
for macro, feature in macros.items():
  DEFAULT_FEATURES.append(
    Feature(name=lambda cfg, m=macro, f=feature: f + ('={}'.format(compilerMacros(cfg)[m]) if compilerMacros(cfg)[m] else ''),
            when=lambda cfg, m=macro: m in compilerMacros(cfg))
  )


# Mapping from canonical locale names (used in the tests) to possible locale
# names on various systems. Each locale is considered supported if any of the
# alternative names is supported.
locales = {
  'en_US.UTF-8':     ['en_US.UTF-8', 'en_US.utf8', 'English_United States.1252'],
  'fr_FR.UTF-8':     ['fr_FR.UTF-8', 'fr_FR.utf8', 'French_France.1252'],
  'ja_JP.UTF-8':     ['ja_JP.UTF-8', 'ja_JP.utf8', 'Japanese_Japan.923'],
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


# Add features representing the target platform name: darwin, linux, windows, etc...
DEFAULT_FEATURES += [
  Feature(name='darwin', when=lambda cfg: '__APPLE__' in compilerMacros(cfg)),
  Feature(name='windows', when=lambda cfg: '_WIN32' in compilerMacros(cfg)),
  Feature(name='windows-dll', when=lambda cfg: '_WIN32' in compilerMacros(cfg) and programSucceeds(cfg, """
            #include <iostream>
            #include <windows.h>
            #include <winnt.h>
            int main(int, char**) {
              // Get a pointer to a data member that gets linked from the C++
              // library. This must be a data member (functions can get
              // thunk inside the calling executable), and must not be
              // something that is defined inline in headers.
              void *ptr = &std::cout;
              // Get a handle to the current main executable.
              void *exe = GetModuleHandle(NULL);
              // The handle points at the PE image header. Navigate through
              // the header structure to find the size of the PE image (the
              // executable).
              PIMAGE_DOS_HEADER dosheader = (PIMAGE_DOS_HEADER)exe;
              PIMAGE_NT_HEADERS ntheader = (PIMAGE_NT_HEADERS)((BYTE *)dosheader + dosheader->e_lfanew);
              PIMAGE_OPTIONAL_HEADER peheader = &ntheader->OptionalHeader;
              void *exeend = (BYTE*)exe + peheader->SizeOfImage;
              // Check if the tested pointer - the data symbol from the
              // C++ library - is located within the exe.
              if (ptr >= exe && ptr <= exeend)
                return 1;
              // Return success if it was outside of the executable, i.e.
              // loaded from a DLL.
              return 0;
            }
          """), actions=[AddCompileFlag('-DTEST_WINDOWS_DLL')]),
  Feature(name='linux', when=lambda cfg: '__linux__' in compilerMacros(cfg)),
  Feature(name='netbsd', when=lambda cfg: '__NetBSD__' in compilerMacros(cfg)),
  Feature(name='freebsd', when=lambda cfg: '__FreeBSD__' in compilerMacros(cfg)),
  Feature(name='LIBCXX-FREEBSD-FIXME', when=lambda cfg: '__FreeBSD__' in compilerMacros(cfg)),
]

# Add features representing the build host platform name.
# The build host could differ from the target platform for cross-compilation.
DEFAULT_FEATURES += [
  Feature(name='buildhost={}'.format(sys.platform.lower().strip())),
  # sys.platform can often be represented by a "sub-system", such as 'win32', 'cygwin', 'mingw', freebsd13 & etc.
  # We define a consolidated feature on a few platforms.
  Feature(name='buildhost=windows', when=lambda cfg: platform.system().lower().startswith('windows')),
  Feature(name='buildhost=freebsd', when=lambda cfg: platform.system().lower().startswith('freebsd')),
  Feature(name='buildhost=aix', when=lambda cfg: platform.system().lower().startswith('aix'))
]

# Detect whether GDB is on the system, has Python scripting and supports
# adding breakpoint commands. If so add a substitution to access it.
def check_gdb(cfg):
  gdb_path = shutil.which('gdb')
  if gdb_path is None:
    return False

  # Check that we can set breakpoint commands, which was added in 8.3.
  # Using the quit command here means that gdb itself exits, not just
  # the "python <...>" command.
  test_src = """\
try:
  gdb.Breakpoint(\"main\").commands=\"foo\"
except AttributeError:
  gdb.execute(\"quit 1\")
gdb.execute(\"quit\")"""

  try:
    stdout = subprocess.check_output(
              [gdb_path, "-ex", "python " + test_src, "--batch"],
              stderr=subprocess.DEVNULL, universal_newlines=True)
  except subprocess.CalledProcessError:
    # We can't set breakpoint commands
    return False

  # Check we actually ran the Python
  return not "Python scripting is not supported" in stdout

DEFAULT_FEATURES += [
  Feature(name='host-has-gdb-with-python',
    when=check_gdb,
    actions=[AddSubstitution('%{gdb}', lambda cfg: shutil.which('gdb'))]
  )
]

# Define features for back-deployment testing.
# These features can be used to XFAIL tests that require features in the dylib that
# were not supported by a certain target. Vendors can tweak how those are defined
# to match their own releases of the dylib.
DEFAULT_FEATURES += [
  # Tests that require std::to_chars(floating-point) in the built library
  Feature(name='availability-fp_to_chars-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx{{(10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0)(.0)?}}', cfg.available_features)),

  # Tests that require https://wg21.link/P0482 support in the built library
  Feature(name='availability-char8_t_support-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx{{(10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0)(.0)?}}', cfg.available_features)),

  # Tests that require __libcpp_verbose_abort support in the built library
  Feature(name='availability-verbose_abort-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx{{(10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0)(.0)?}}', cfg.available_features)),

  # Tests that require std::bad_variant_access in the built library
  Feature(name='availability-bad_variant_access-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx10.{{(9|10|11|12)(.0)?}}', cfg.available_features)),

  # Tests that require std::bad_optional_access in the built library
  Feature(name='availability-bad_optional_access-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx10.{{(9|10|11|12)(.0)?}}', cfg.available_features)),

  # Tests that require std::bad_any_cast in the built library
  Feature(name='availability-bad_any_cast-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx10.{{(9|10|11|12)(.0)?}}', cfg.available_features)),

  # Tests that require std::pmr support in the built library
  Feature(name='availability-pmr-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx{{(10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0)(.0)?}}', cfg.available_features)),

  # Tests that require std::filesystem support in the built library
  Feature(name='availability-filesystem-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx10.{{(9|10|11|12|13|14)(.0)?}}', cfg.available_features)),

  # Tests that require the C++20 synchronization library (P1135R6 implemented by https://llvm.org/D68480) in the built library
  Feature(name='availability-synchronization_library-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx10.{{(9|10|11|12|13|14|15)(.0)?}}', cfg.available_features)),

  # Tests that require support for std::shared_mutex and std::shared_timed_mutex in the built library
  Feature(name='availability-shared_mutex-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx10.{{(9|10|11)(.0)?}}', cfg.available_features)),

  # Tests that require support for aligned allocation in the built library. This is about `operator new(..., std::align_val_t, ...)` specifically,
  # not other forms of aligned allocation.
  Feature(name='availability-aligned_allocation-missing',
    when=lambda cfg: BooleanExpression.evaluate('use_system_cxx_lib && target={{.+}}-apple-macosx10.{{(9|10|11|12)(.0)?}}', cfg.available_features)),
]
