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

# RUN: %{python} %s %{libcxx-dir}/utils

# block Lit from interpreting a RUN/XFAIL/etc inside the generation script
# END.

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import lit_header_restrictions, public_headers

for header in public_headers:
  print(f"""\
//--- {header}.compile.pass.cpp
// RUN: %{{cxx}} %s %{{flags}} %{{compile_flags}} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only

// GCC doesn't support -fcxx-modules
// UNSUPPORTED: gcc

// The Windows headers don't appear to be compatible with modules
// UNSUPPORTED: windows
// UNSUPPORTED: buildhost=windows

// The Android headers don't appear to be compatible with modules yet
// UNSUPPORTED: LIBCXX-ANDROID-FIXME

// TODO: Investigate this failure
// UNSUPPORTED: LIBCXX-FREEBSD-FIXME

{lit_header_restrictions.get(header, '')}

#include <{header}>
""")

print(f"""\
//--- __std_clang_module.compile.pass.mm
// RUN: %{{cxx}} %s %{{flags}} %{{compile_flags}} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only

// REQUIRES: clang-modules-build

// GCC doesn't support -fcxx-modules
// UNSUPPORTED: gcc

// The Windows headers don't appear to be compatible with modules
// UNSUPPORTED: windows
// UNSUPPORTED: buildhost=windows

// The Android headers don't appear to be compatible with modules yet
// UNSUPPORTED: LIBCXX-ANDROID-FIXME

// TODO: Investigate this failure
// UNSUPPORTED: LIBCXX-FREEBSD-FIXME

@import std;

""")
