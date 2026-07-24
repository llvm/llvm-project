//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++14

// Check that functions are marked [[nodiscard]]

#include <type_traits>

#include "test_macros.h"

void test() {
  std::true_type tag;
  tag(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 20
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_constant_evaluated();
#endif
}

#if TEST_STD_VER >= 26 && defined(__cpp_lib_is_within_lifetime)
consteval void test_consteval() {
  [[maybe_unused]] int n = 0;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_within_lifetime(&n);
}
#endif
