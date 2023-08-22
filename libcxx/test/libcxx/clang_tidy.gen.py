#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Run our custom libc++ clang-tidy checks on all public headers.

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import lit_header_restrictions, public_headers

for header in public_headers:
  BLOCKLIT = '' # block Lit from interpreting a RUN/XFAIL/etc inside the generation script
  print(f"""\
//--- {header}.sh.cpp

// REQUIRES{BLOCKLIT}: has-clang-tidy

// The GCC compiler flags are not always compatible with clang-tidy.
// UNSUPPORTED{BLOCKLIT}: gcc

{lit_header_restrictions.get(header, '')}

// TODO: run clang-tidy with modules enabled once they are supported
// RUN{BLOCKLIT}: %{{clang-tidy}} %s --warnings-as-errors=* -header-filter=.* --checks='-*,libcpp-*' --load=%{{test-tools}}/clang_tidy_checks/libcxx-tidy.plugin -- %{{compile_flags}} -fno-modules
// RUN{BLOCKLIT}: %{{clang-tidy}} %s --warnings-as-errors=* -header-filter=.* --config-file=%{{libcxx}}/.clang-tidy -- -Wweak-vtables %{{compile_flags}} -fno-modules

#include <{header}>
""")
