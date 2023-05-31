#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Test that headers are not tripped up by the surrounding code defining the
# min() and max() macros.

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.test.header_information import header_restrictions, public_headers

for header in public_headers:
  test_condition_begin = '#if ' + header_restrictions[header] if header in header_restrictions else ''
  test_condition_end = '#endif' if header in header_restrictions else ''

  print(f"""\
//--- {header}.compile.pass.cpp
#define TEST_MACROS() static_assert(min() == true && max() == true, "")
#define min() true
#define max() true

#include <__config>
{test_condition_begin}
#include <{header}>
TEST_MACROS();
{test_condition_end}
""")
