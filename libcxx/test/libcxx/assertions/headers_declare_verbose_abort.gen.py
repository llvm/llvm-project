#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Test that all public C++ headers define the verbose termination function, which
# is required for users to be able to include any public header and then override
# the function using a strong definition.

# RUN: %{python} %s %{libcxx}/utils

import sys
sys.path.append(sys.argv[1])
from libcxx.test.header_information import header_restrictions, public_headers

for header in public_headers:
    # Skip C compatibility headers.
    if header.endswith('.h'):
        continue

    test_condition_begin = f'#if {header_restrictions[header]}' if header in header_restrictions else ''
    test_condition_end = '#endif' if header in header_restrictions else ''
    XFAIL = 'XFAIL' # Make sure Lit doesn't think we are XFAILing this test
    print(f"""\
//--- {header}.compile.pass.cpp
// {XFAIL}: availability-verbose_abort-missing
#include <__config>
{test_condition_begin}
#include <{header}>
using HandlerType = decltype(std::__libcpp_verbose_abort);
{test_condition_end}
""")
