# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Test that we can include each header in a TU while using modules.
# This is important notably because the LLDB data formatters use
# libc++ headers with modules enabled.

# Older macOS SDKs were not properly modularized, which causes issues with localization.
# This feature should instead be based on the SDK version.
# UNSUPPORTED: stdlib=system && target={{.+}}-apple-macosx13{{.*}}

# GCC doesn't support -fcxx-modules
# UNSUPPORTED: gcc

# The Windows headers don't appear to be compatible with modules
# UNSUPPORTED: windows
# UNSUPPORTED: buildhost=windows

# The Android headers don't appear to be compatible with modules yet
# UNSUPPORTED: LIBCXX-ANDROID-FIXME

# TODO: Investigate this failure
# UNSUPPORTED: LIBCXX-FREEBSD-FIXME

# TODO: Investigate why this doesn't work on Picolibc once the locale base API is refactored
# UNSUPPORTED: LIBCXX-PICOLIBC-FIXME

# TODO: Fix seemingly circular inclusion or <wchar.h> on AIX
# UNSUPPORTED: LIBCXX-AIX-FIXME

# UNSUPPORTED: FROZEN-CXX03-HEADERS-FIXME

# RUN: %{python} %s %{libcxx-dir}/utils
# END.

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import (
    lit_header_restrictions,
    lit_header_undeprecations,
    public_headers,
)

for header in public_headers:
    print(
        f"""\
//--- {header}.compile.pass.cpp
// RUN: %{{cxx}} %s %{{flags}} %{{compile_flags}} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only

{lit_header_restrictions.get(header, '')}
{lit_header_undeprecations.get(header, '')}

#include <{header}>
"""
    )

print(
    f"""\
//--- import_std.compile.pass.mm
// RUN: %{{cxx}} %s %{{flags}} %{{compile_flags}} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only

// REQUIRES: clang-modules-build

@import std;
"""
)
