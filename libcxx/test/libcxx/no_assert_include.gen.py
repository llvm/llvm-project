# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Ensure that none of the standard C++ headers implicitly include cassert or
# assert.h (because assert() is implemented as a macro).

# RUN: %{python} %s %{libcxx-dir}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import lit_header_restrictions, public_headers

for header in public_headers:
  if header == 'cassert':
    continue

  print(f"""\
//--- {header}.compile.pass.cpp
{lit_header_restrictions.get(header, '')}

#include <{header}>

#ifdef assert
# error "Do not include cassert or assert.h in standard header files"
#endif
""")
