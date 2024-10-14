//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cstdbool>

// check that <cstdbool> is deprecated in C++17 and removed in C++20

// UNSUPPORTED: c++03, c++11, c++14

#include "test_macros.h"

#if TEST_STD_VER >= 20
#  include <cstdbool>
// expected-warning {{'__standard_header_cstdbool' is deprecated: removed in C++20}}
#else
#  include <cstdbool>
// expected-warning {{'__standard_header_cstdbool' is deprecated}}
#endif
