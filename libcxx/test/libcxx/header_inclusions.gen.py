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
from libcxx.header_information import lit_header_restrictions, public_headers, mandatory_inclusions

for header in public_headers:
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
{lit_header_restrictions.get(header, '')}

#include <{header}>
{checks}
""")
