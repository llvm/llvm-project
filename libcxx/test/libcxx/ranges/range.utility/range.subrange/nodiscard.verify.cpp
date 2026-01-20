//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test that functions are marked [[nodiscard]].

#include <ranges>
#include <utility>
#include <vector>

#include "test_range.h"

void test() {
  std::vector<int> range;
  std::ranges::subrange subrange{range.begin(), range.end()};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(subrange).begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  subrange.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(subrange).end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(subrange).empty();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(subrange).size();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(subrange).next();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(subrange).next();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(subrange).prev();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(subrange).prev();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::as_const(subrange));
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::get<0>(std::move(subrange));
}
