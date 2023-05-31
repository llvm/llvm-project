#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Test that all headers include all the other headers they're supposed to, as
# prescribed by the Standard.

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.test.header_information import header_restrictions, public_headers, mandatory_inclusions

for header in public_headers:
  test_condition_begin = '#if ' + header_restrictions[header] if header in header_restrictions else ''
  test_condition_end = '#endif' if header in header_restrictions else ''

  header_guard = lambda h: f"_LIBCPP_{h.upper().replace('.', '_').replace('/', '_')}"

  # <cassert> has no header guards
  if header == 'cassert':
    checks = ''
  else:
    checks = f'''
#ifndef {header_guard(header)}
# error <{header}> was expected to define a header guard {header_guard(header)}
#endif
'''
  for includee in mandatory_inclusions.get(header, []):
    checks += f'''
#ifndef {header_guard(includee)}
# error <{header}> was expected to include <{includee}>
#endif
'''

  print(f"""\
//--- {header}.compile.pass.cpp
#include <__config>
{test_condition_begin}
#include <{header}>
{checks}
{test_condition_end}
""")
