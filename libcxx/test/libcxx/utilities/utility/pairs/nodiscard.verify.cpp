//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that functions are marked [[nodiscard]]

#include <utility>

#include <test_macros.h>

void test() {
  struct First {};
  struct Second {};

  std::pair<First, Second> p;
  const std::pair<First, Second> cp;

  std::make_pair(94, 82); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 11
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(p);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(cp);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(p));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(cp));
#endif // TEST_STD_VER >= 11
#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<First>(p);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<First>(cp);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<First>(std::move(p));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<First>(std::move(cp));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Second>(p);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Second>(cp);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Second>(std::move(p));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Second>(std::move(cp));
#endif // TEST_STD_VER >= 14
}
