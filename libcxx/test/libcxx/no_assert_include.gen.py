#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Ensure that none of the standard C++ headers implicitly include cassert or
# assert.h (because assert() is implemented as a macro).

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.test.header_information import header_restrictions, public_headers

for header in public_headers:
  if header == 'cassert':
    continue

  test_condition_begin = '#if ' + header_restrictions[header] if header in header_restrictions else ''
  test_condition_end = '#endif' if header in header_restrictions else ''

  print(f"""\
//--- {header}.compile.pass.cpp

#include <__config>
{test_condition_begin}
#include <{header}>
#ifdef assert
# error "Do not include cassert or assert.h in standard header files"
#endif
{test_condition_end}
""")
