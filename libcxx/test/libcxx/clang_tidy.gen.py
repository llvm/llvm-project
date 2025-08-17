# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##


# Run our custom libc++ clang-tidy checks on all public headers.

# RUN: %{python} %s %{libcxx-dir}/utils

# block Lit from interpreting a RUN/XFAIL/etc inside the generation script
# END.

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import lit_header_restrictions, lit_header_undeprecations, public_headers

for header in public_headers:
  print(f"""\
//--- {header}.sh.cpp

// REQUIRES: has-clang-tidy

// The frozen headers should not be updated to the latest libc++ style, so don't test.
// UNSUPPORTED: FROZEN-CXX03-HEADERS-FIXME

// The GCC compiler flags are not always compatible with clang-tidy.
// UNSUPPORTED: gcc

{lit_header_restrictions.get(header, '')}
{lit_header_undeprecations.get(header, '')}

// TODO: run clang-tidy with modules enabled once they are supported
// RUN: %{{clang-tidy}} %s --warnings-as-errors=* -header-filter=.* --config-file=%{{libcxx-dir}}/.clang-tidy --load=%{{test-tools-dir}}/clang_tidy_checks/libcxx-tidy.plugin -- -Wweak-vtables %{{compile_flags}} -fno-modules

#include <{header}>
""")
