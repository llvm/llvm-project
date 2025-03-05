//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <limits> functions are marked [[nodiscard]]

#include <limits>

#include "test_macros.h"

void func() {
  // clang-format off
  // arithmetic
  std::numeric_limits<int>::min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::lowest(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::epsilon(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::round_error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::infinity(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::quiet_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::signaling_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<int>::denorm_min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // bool
  std::numeric_limits<bool>::min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::lowest(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::epsilon(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::round_error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::infinity(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::quiet_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::signaling_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<bool>::denorm_min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // float
  std::numeric_limits<float>::min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::lowest(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::epsilon(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::round_error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::infinity(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::quiet_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::signaling_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<float>::denorm_min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // double
  std::numeric_limits<double>::min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::lowest(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::epsilon(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::round_error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::infinity(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::quiet_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::signaling_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<double>::denorm_min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // long double
  std::numeric_limits<long double>::min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::max(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::lowest(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::epsilon(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::round_error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::infinity(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::quiet_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::signaling_NaN(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::numeric_limits<long double>::denorm_min(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}
