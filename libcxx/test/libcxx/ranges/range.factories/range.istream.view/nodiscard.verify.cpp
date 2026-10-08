//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-localization

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <sstream>
#include <utility>

void test() {
  std::stringstream ss;
  auto v = std::views::istream<char>(ss);

  // [range.istream.view]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).end();

  // [range.istream.iterator]

  auto it = v.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *it;

  // [range.istream.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::istream<char>(ss);
}
