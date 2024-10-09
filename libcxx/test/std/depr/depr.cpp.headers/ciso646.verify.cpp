//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ciso646>

// check that <ciso646> is removed in C++20

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include "test_macros.h"

// expected-warning {{'__standard_header_ciso646' is deprecated: removed in C++20}}

#include <ciso646>
