//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that stride_view's member functions are properly marked nodiscard.

#include <ranges>
#include <utility>

void test() {
  int range[] = {1, 2, 3};
  auto sv     = std::ranges::stride_view(range, 2);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(sv).base();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.stride();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::stride(range, 2);
}
