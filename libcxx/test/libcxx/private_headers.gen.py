#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# This test ensures that we produce a diagnostic when we use a private header
# from user code.

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.test.header_information import lit_header_restrictions, private_headers, private_headers_still_public_in_modules

for header in private_headers:
  # Skip headers that are not private yet in the modulemap
  if header in private_headers_still_public_in_modules:
    continue

  # Skip private headers that start with __support -- those are not in the modulemap yet
  if header.startswith('__support'):
    continue

  # Skip the locale API headers, since they are platform-specific and thus inherently non-modular
  if 'locale_base_api' in header:
    continue

  # TODO: Stop skipping PSTL headers once their integration is finished.
  if header.startswith('__pstl'):
    continue

  BLOCKLIT = '' # block Lit from interpreting a RUN/XFAIL/etc inside the generation script
  print(f"""\
//--- {header}.verify.cpp
// REQUIRES{BLOCKLIT}: modules-build
{lit_header_restrictions.get(header, '')}

#include <{header}> // expected-error@*:* {{{{use of private header from outside its module: '{header}'}}}}
""")
