#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Test that all headers define the _LIBCPP_VERSION macro.

# RUN: %{python} %s %{libcxx-dir}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.header_information import lit_header_restrictions, public_headers

for header in public_headers:
  print(f"""\
//--- {header}.compile.pass.cpp
{lit_header_restrictions.get(header, '')}

#include <{header}>

#ifndef _LIBCPP_VERSION
# error <{header}> does not seem to define _LIBCPP_VERSION
#endif
""")
