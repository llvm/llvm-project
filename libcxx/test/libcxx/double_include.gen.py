#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Test that we can include each header in two TU's and link them together.

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.test.header_information import header_restrictions, public_headers

BLOCKLIT = '' # block Lit from interpreting a RUN/XFAIL/etc inside the generation script
print(f"""\
//--- double_include.sh.cpp
// RUN{BLOCKLIT}: %{{cxx}} -c %s -o %t.first.o %{{flags}} %{{compile_flags}}
// RUN{BLOCKLIT}: %{{cxx}} -c %s -o %t.second.o -DWITH_MAIN %{{flags}} %{{compile_flags}}
// RUN{BLOCKLIT}: %{{cxx}} -o %t.exe %t.first.o %t.second.o %{{flags}} %{{link_flags}}
// RUN{BLOCKLIT}: %{{run}}
""")

for header in public_headers:
  test_condition_begin = '#if ' + header_restrictions[header] if header in header_restrictions else ''
  test_condition_end = '#endif' if header in header_restrictions else ''
  print(f"""\
#include <__config>
{test_condition_begin}
#include <{header}>
{test_condition_end}
""")

print("""
#if defined(WITH_MAIN)
int main(int, char**) { return 0; }
#endif
""")
