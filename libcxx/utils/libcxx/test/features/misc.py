# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import compilerMacros, sourceBuilds, hasCompileFlag, programSucceeds, runScriptExitCode
from libcxx.test.dsl import Feature, AddCompileFlag, AddLinkFlag
import platform
import sys

def _mingwSupportsModules(cfg):
    # Only mingw headers are known to work with libc++ built as a module,
    # at the moment.
    if not "__MINGW32__" in compilerMacros(cfg):
        return False
    # For mingw headers, check for a version known to support being built
    # as a module.
    return sourceBuilds(
        cfg,
        """
        #include <_mingw_mac.h>
        #if __MINGW64_VERSION_MAJOR < 12
        #error Headers known to be incompatible
        #elif __MINGW64_VERSION_MAJOR == 12
        // The headers were fixed to work with libc++ modules during
        // __MINGW64_VERSION_MAJOR == 12. The headers became compatible
        // with libc++ built as a module in
        // 1652e9241b5d8a5a779c6582b1c3c4f4a7cc66e5 (Apr 2024), but the
        // following commit 8c13b28ace68f2c0094d45121d59a4b951b533ed
        // removed the now unused __mingw_static_ovr define. Use this
        // as indicator for whether we've got new enough headers.
        #ifdef __mingw_static_ovr
        #error Headers too old
        #endif
        #else
        // __MINGW64_VERSION_MAJOR > 12 should be ok.
        #endif
        int main(int, char**) { return 0; }
        """,
    )

features = [
    Feature(
        name="diagnose-if-support",
        when=lambda cfg: hasCompileFlag(cfg, "-Wuser-defined-warnings"),
        actions=[AddCompileFlag("-Wuser-defined-warnings")],
    ),
    Feature(
        name="character-conversion-warnings",
        when=lambda cfg: hasCompileFlag(cfg, "-Wcharacter-conversion"),
    ),
    # Tests to validate whether the compiler has a way to set the maximum number
    # of steps during constant evaluation. Since the flag differs per compiler
    # store the "valid" flag as a feature. This allows passing the proper compile
    # flag to the compiler:
    # // ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12345678
    # // ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=12345678
    Feature(
        name="has-fconstexpr-steps",
        when=lambda cfg: hasCompileFlag(cfg, "-fconstexpr-steps=1"),
    ),
    Feature(
        name="has-fconstexpr-ops-limit",
        when=lambda cfg: hasCompileFlag(cfg, "-fconstexpr-ops-limit=1"),
    ),
    Feature(name="has-fblocks", when=lambda cfg: hasCompileFlag(cfg, "-fblocks")),
    Feature(
        name="fdelayed-template-parsing",
        when=lambda cfg: hasCompileFlag(cfg, "-fdelayed-template-parsing"),
    ),
    Feature(
        name="has-fobjc-arc",
        when=lambda cfg: hasCompileFlag(cfg, "-xobjective-c++ -fobjc-arc")
        and sys.platform.lower().strip() == "darwin",
    ),  # TODO: this doesn't handle cross-compiling to Apple platforms.
    Feature(
        name="objective-c++",
        when=lambda cfg: hasCompileFlag(cfg, "-xobjective-c++ -fobjc-arc"),
    ),
    Feature(
        name="verify-support",
        when=lambda cfg: hasCompileFlag(cfg, "-Xclang -verify-ignore-unexpected"),
    ),
    Feature(
        name="add-latomic-workaround",  # https://llvm.org/PR73361
        when=lambda cfg: sourceBuilds(
            cfg, "int main(int, char**) { return 0; }", ["-latomic"]
        ),
        actions=[AddLinkFlag("-latomic")],
    ),
    Feature(
        name="has-64-bit-atomics",
        when=lambda cfg: sourceBuilds(
            cfg,
            """
            #include <atomic>
            struct Large { char storage[64/8]; };
            std::atomic<Large> x;
            int main(int, char**) { (void)x.load(); (void)x.is_lock_free(); return 0; }
          """,
        ),
    ),
    Feature(
        name="has-1024-bit-atomics",
        when=lambda cfg: sourceBuilds(
            cfg,
            """
            #include <atomic>
            struct Large { char storage[1024/8]; };
            std::atomic<Large> x;
            int main(int, char**) { (void)x.load(); (void)x.is_lock_free(); return 0; }
          """,
        ),
    ),
    # Tests that require 64-bit architecture
    Feature(
        name="32-bit-pointer",
        when=lambda cfg: sourceBuilds(
            cfg,
            """
            int main(int, char**) {
              static_assert(sizeof(void *) == 4);
            }
          """,
        ),
    ),
    # Check for a Windows UCRT bug (fixed in UCRT/Windows 10.0.20348.0):
    # https://developercommunity.visualstudio.com/t/utf-8-locales-break-ctype-functions-for-wchar-type/1653678
    Feature(
        name="win32-broken-utf8-wchar-ctype",
        when=lambda cfg: not "_LIBCPP_HAS_LOCALIZATION" in compilerMacros(cfg)
        or compilerMacros(cfg)["_LIBCPP_HAS_LOCALIZATION"] == "1"
        and "_WIN32" in compilerMacros(cfg)
        and not programSucceeds(
            cfg,
            """
            #include <locale.h>
            #include <wctype.h>
            int main(int, char**) {
              setlocale(LC_ALL, "en_US.UTF-8");
              return towlower(L'\\xDA') != L'\\xFA';
            }
          """,
        ),
    ),
    # Check for a Windows UCRT bug (fixed in UCRT/Windows 10.0.19041.0).
    # https://developercommunity.visualstudio.com/t/printf-formatting-with-g-outputs-too/1660837
    Feature(
        name="win32-broken-printf-g-precision",
        when=lambda cfg: "_WIN32" in compilerMacros(cfg)
        and not programSucceeds(
            cfg,
            """
            #include <stdio.h>
            #include <string.h>
            int main(int, char**) {
              char buf[100];
              snprintf(buf, sizeof(buf), "%#.*g", 0, 0.0);
              return strcmp(buf, "0.");
            }
          """,
        ),
    ),
    # Check for a Windows UCRT bug (not fixed upstream yet).
    # With UCRT, printf("%a", 0.0) produces "0x0.0000000000000p+0",
    # while other C runtimes produce just "0x0p+0".
    # https://developercommunity.visualstudio.com/t/Printf-formatting-of-float-as-hex-prints/1660844
    Feature(
        name="win32-broken-printf-a-precision",
        when=lambda cfg: "_WIN32" in compilerMacros(cfg)
        and not programSucceeds(
            cfg,
            """
            #include <stdio.h>
            #include <string.h>
            int main(int, char**) {
              char buf[100];
              snprintf(buf, sizeof(buf), "%a", 0.0);
              return strcmp(buf, "0x0p+0");
            }
          """,
        ),
    ),
    Feature(
        name="has-unix-headers",
        when=lambda cfg: sourceBuilds(
            cfg,
            """
            #include <unistd.h>
            #include <sys/wait.h>
            int main(int, char**) {
              int fd[2];
              return pipe(fd);
            }
          """,
        ),
    ),
    # Whether Bash can run on the executor.
    # This is not always the case, for example when running on embedded systems.
    #
    # For the corner case of bash existing, but it being missing in the path
    # set in %{exec} as "--env PATH=one-single-dir", the executor does find
    # and executes bash, but bash then can't find any other common shell
    # utilities. Test executing "bash -c 'bash --version'" to see if bash
    # manages to find binaries to execute.
    Feature(
        name="executor-has-no-bash",
        when=lambda cfg: runScriptExitCode(cfg, ["%{exec} bash -c 'bash --version'"]) != 0,
    ),
    # Whether module support for the platform is available.
    Feature(
        name="has-no-cxx-module-support",
        # The libc of these platforms have functions with internal linkage.
        # This is not allowed per C11 7.1.2 Standard headers/6
        #  Any declaration of a library function shall have external linkage.
        when=lambda cfg: "__ANDROID__" in compilerMacros(cfg)
        or "__FreeBSD__" in compilerMacros(cfg)
        or ("_WIN32" in compilerMacros(cfg) and not _mingwSupportsModules(cfg))
        or platform.system().lower().startswith("aix")
        # Avoid building on platforms that don't support modules properly.
        or not hasCompileFlag(cfg, "-Wno-reserved-module-identifier")
        # older versions don't support extern "C++", newer versions don't support main in named module.
        or not (
            sourceBuilds(
                cfg,
                """
            export module test;
            extern "C++" int main(int, char**) { return 0; }
          """,
            )
            or sourceBuilds(
                cfg,
                """
            export module test;
            int main(int, char**) { return 0; }
          """,
            )
        ),
    ),
    # The time zone validation tests compare the output of zdump against the
    # output generated by <chrono>'s time zone support.
    Feature(
        name="has-no-zdump",
        when=lambda cfg: runScriptExitCode(cfg, ["zdump --version"]) != 0,
    ),
    Feature(
        name="can-create-symlinks",
        when=lambda cfg: "_WIN32" not in compilerMacros(cfg)
        or programSucceeds(
            cfg,
            # Creation of symlinks require elevated privileges on Windows unless
            # Windows developer mode is enabled.
            """
            #include <stdio.h>
            #include <windows.h>
            int main(int, char**) {
              CHAR tempDirPath[MAX_PATH];
              DWORD tempPathRet = GetTempPathA(MAX_PATH, tempDirPath);
              if (tempPathRet == 0 || tempPathRet > MAX_PATH) {
                return 1;
              }

              CHAR tempFilePath[MAX_PATH];
              UINT uRetVal = GetTempFileNameA(
                tempDirPath,
                "cxx", // Prefix
                0, // Unique=0 also implies file creation.
                tempFilePath);
              if (uRetVal == 0) {
                return 1;
              }

              CHAR symlinkFilePath[MAX_PATH];
              int ret = sprintf_s(symlinkFilePath, MAX_PATH, "%s_symlink", tempFilePath);
              if (ret == -1) {
                DeleteFileA(tempFilePath);
                return 1;
              }

              // Requires either administrator, or developer mode enabled.
              BOOL bCreatedSymlink = CreateSymbolicLinkA(symlinkFilePath,
                tempFilePath,
                SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE);
              if (!bCreatedSymlink) {
                DeleteFileA(tempFilePath);
                return 1;
              }

              DeleteFileA(tempFilePath);
              DeleteFileA(symlinkFilePath);
              return 0;
            }
            """,
        ),
    ),
]
