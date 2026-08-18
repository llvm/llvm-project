//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// These compilers do not support __builtin_type_order
// UNSUPPORTED: clang-21, clang-22, clang-23, apple-clang-21
// UNSUPPORTED: gcc-15

// Check that type_order::operator() is marked [[nodiscard]]

#include <compare>

void test() {
  std::type_order<int, char>()();
  // expected-warning@-1 {{ignoring return value of function declared with 'nodiscard' attribute}}
}
