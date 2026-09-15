//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that stride_view's iterator member functions are properly marked nodiscard.

#include <ranges>
#include <utility>

void test() {
  int range[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto sv     = std::ranges::stride_view(std::ranges::ref_view(range), 3);
  auto it     = sv.begin();
  auto it2    = sv.begin();
  ++it2;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(it).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::as_const(it);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it2 - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it2 - it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::default_sentinel_t() - it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - std::default_sentinel_t();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::iter_move(it);
}
