//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11

// Check that functions are marked [[nodiscard]]

#include <tuple>

#include "test_macros.h"

void test() {
  struct First {};
  struct Second {};
  struct Third {};

  std::tuple<First, Second, Third> t;
  const std::tuple<First, Second, Third> ct;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(t);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(ct);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(t));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(t));
#if TEST_STD_VER >= 14
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Third>(t);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Third>(ct);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Third>(std::move(t));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<Third>(std::move(t));
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tie(ct);

  First e1;
  Second e2;
  Third e3;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_tuple(e1, e2, e3);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::forward_as_tuple(First{}, Second{}, Third{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tuple_cat();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::tuple_cat(std::tuple<First>{}, std::tuple<Second, Third>{});

#if TEST_STD_VER >= 17
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_from_tuple<First>(std::tuple<First>{});
#endif
}
