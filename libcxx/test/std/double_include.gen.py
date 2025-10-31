# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Test that we can include each header in two TU's and link them together.

# We're using compiler-specific flags in this test
# REQUIRES: (gcc || clang)

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
//--- {header}.sh.cpp
{lit_header_restrictions.get(header, '')}
{lit_header_undeprecations.get(header, '')}

// RUN: %{{cxx}} -c %s -o %t.first.o %{{flags}} %{{compile_flags}}
// RUN: %{{cxx}} -c %s -o %t.second.o -DWITH_MAIN %{{flags}} %{{compile_flags}}
// RUN: %{{cxx}} -o %t.exe %t.first.o %t.second.o %{{flags}} %{{link_flags}}
// RUN: %{{run}}

#include <{header}>

#if defined(WITH_MAIN)
int main(int, char**) {{ return 0; }}
#endif
"""
    )
