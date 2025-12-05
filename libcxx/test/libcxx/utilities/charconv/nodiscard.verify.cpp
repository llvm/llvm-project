//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <flat_set>

// Check that functions are marked [[nodiscard]]

#include <charconv>

void test() {
  char buf[32];

  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_chars(buf, buf + sizeof(buf), 49.0f);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_chars(buf, buf + sizeof(buf), 82.0);

  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_chars(buf, buf + sizeof(buf), 49);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::to_chars(buf, buf + sizeof(buf), 49, 16);

  float f2;
  double d2;
  long double ld2;

  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), f2);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), d2);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), ld2);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), f2, std::chars_format::general);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), d2, std::chars_format::general);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), ld2, std::chars_format::general);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), f2, std::chars_format::general, 5);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), d2, std::chars_format::general, 5);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), ld2, std::chars_format::general, 5);

  int i;

  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), i);
  // expected-warning @+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::from_chars(buf, buf + sizeof(buf), i, 16);
}