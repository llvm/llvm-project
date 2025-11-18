//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ctgmath>

// check that <ctgmath> is deprecated in C++17 and removed in C++20
// When built with modules, <ctgmath> should be omitted.

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: clang-modules-build

#include <ctgmath>

// expected-warning@ctgmath:* {{<ctgmath> is deprecated in C++17 and removed in C++20. Include <cmath> and <complex> instead.}}
