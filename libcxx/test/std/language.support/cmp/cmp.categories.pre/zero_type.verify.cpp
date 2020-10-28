// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// Ensure we reject all cases where an argument other than a literal 0 is used
// for a comparison against a comparison category type.

#include <compare>

#define TEST_OP(v, op)                                                         \
  void(v op 0L);                                                               \
  void(0L op v);                                                               \
  void(v op nullptr);                                                          \
  void(nullptr op v);                                                          \
  void(v op(1 - 1));                                                           \
  void((1 - 1) op v);

template <typename T>
void test_category(T v) {
  TEST_OP(v, ==);  // expected-error 18 {{}}
  TEST_OP(v, !=);  // expected-error 18 {{}}
  TEST_OP(v, <);   // expected-error 18 {{}}
  TEST_OP(v, <=);  // expected-error 18 {{}}
  TEST_OP(v, >);   // expected-error 18 {{}}
  TEST_OP(v, >=);  // expected-error 18 {{}}
  TEST_OP(v, <=>); // expected-error 18 {{}}

  void(v == 0);
  void(0 == v);
  void(v != 0);
  void(0 != v);
  void(v < 0);
  void(0 < v);
  void(v <= 0);
  void(0 <= v);
  void(v > 0);
  void(0 > v);
  void(v >= 0);
  void(0 >= v);
#ifndef _LIBCPP_HAS_NO_SPACESHIP_OPERATOR
  void(v <=> 0);
  void(0 <=> v);
#endif
}

int main(int, char**) {
  test_category(std::strong_ordering::equivalent);
  test_category(std::weak_ordering::equivalent);
  test_category(std::partial_ordering::equivalent);
  return 0;
}
