//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <ranges>

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <utility>

void test() {
  auto v = std::views::single(94);

  // [range.single.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.empty();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.data();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).data();

  // [range.single.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::single(82);
}
