# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

from libcxx.test.dsl import programOutput, Feature, compilerMacros, programSucceeds, AddCompileFlag, sourceBuilds
import platform
import sys

def _getAndroidDeviceApi(cfg):
    return int(
        programOutput(
            cfg,
            r"""
                #include <android/api-level.h>
                #include <stdio.h>
                int main(int, char**) {
                    printf("%d\n", android_get_device_api_level());
                    return 0;
                }
            """,
        )
    )

# Add features representing the target platform name: darwin, linux, windows, etc...
features = [
    Feature(name="darwin", when=lambda cfg: "__APPLE__" in compilerMacros(cfg)),
    Feature(name="windows", when=lambda cfg: "_WIN32" in compilerMacros(cfg)),
    Feature(
        name="windows-dll",
        when=lambda cfg: "_WIN32" in compilerMacros(cfg)
        and sourceBuilds(
            cfg,
            """
            #include <iostream>
            int main(int, char**) { return 0; }
          """,
        )
        and programSucceeds(
            cfg,
            """
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
          """,
        ),
        actions=[AddCompileFlag("-DTEST_WINDOWS_DLL")],
    ),
    Feature(name="linux", when=lambda cfg: "__linux__" in compilerMacros(cfg)),
    Feature(name="android", when=lambda cfg: "__ANDROID__" in compilerMacros(cfg)),
    Feature(
        name=lambda cfg: "android-device-api={}".format(_getAndroidDeviceApi(cfg)),
        when=lambda cfg: "__ANDROID__" in compilerMacros(cfg),
    ),
    Feature(
        name="LIBCXX-ANDROID-FIXME",
        when=lambda cfg: "__ANDROID__" in compilerMacros(cfg),
    ),
    Feature(name="netbsd", when=lambda cfg: "__NetBSD__" in compilerMacros(cfg)),
    Feature(name="freebsd", when=lambda cfg: "__FreeBSD__" in compilerMacros(cfg)),
    Feature(
        name="LIBCXX-FREEBSD-FIXME",
        when=lambda cfg: "__FreeBSD__" in compilerMacros(cfg),
    ),
    Feature(
        name="LIBCXX-PICOLIBC-FIXME",
        when=lambda cfg: sourceBuilds(
            cfg,
            """
            #include <string.h>
            #ifndef __PICOLIBC__
            #error not picolibc
            #endif
            int main(int, char**) { return 0; }
          """,
        ),
    ),
    Feature(
        name="LIBCXX-AMDGPU-FIXME",
        when=lambda cfg: "__AMDGPU__" in compilerMacros(cfg),
    ),
    Feature(
        name="LIBCXX-NVPTX-FIXME",
        when=lambda cfg: "__NVPTX__" in compilerMacros(cfg),
    ),
]

# Add features representing the build host platform name.
# The build host could differ from the target platform for cross-compilation.
features += [
    Feature(name="buildhost={}".format(sys.platform.lower().strip())),
    # sys.platform can often be represented by a "sub-system", such as 'win32', 'cygwin', 'mingw', freebsd13 & etc.
    # We define a consolidated feature on a few platforms.
    Feature(
        name="buildhost=windows",
        when=lambda cfg: platform.system().lower().startswith("windows"),
    ),
    Feature(
        name="buildhost=freebsd",
        when=lambda cfg: platform.system().lower().startswith("freebsd"),
    ),
    Feature(
        name="buildhost=aix",
        when=lambda cfg: platform.system().lower().startswith("aix"),
    ),
]
