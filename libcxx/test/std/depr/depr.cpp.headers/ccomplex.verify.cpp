//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ccomplex>

// check that <ccomplex> is deprecated in C++17 and removed in C++20
// When built with modules, <ccomplex> should be omitted.

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: clang-modules-build

#include "test_macros.h"

#include <ccomplex>

#if TEST_STD_VER >= 20
// expected-warning@ccomplex:* {{'__standard_header_ccomplex' is deprecated: removed in C++20. Include <complex> instead.}}
#else
// expected-warning@ccomplex:* {{'__standard_header_ccomplex' is deprecated: Include <complex> instead.}}
#endif
