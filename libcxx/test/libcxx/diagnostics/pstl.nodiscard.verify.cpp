//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that PSTL algorithms are marked [[nodiscard]] as a conforming extension

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// UNSUPPORTED: c++03, c++11, c++14

// clang-format off

#include <algorithm>
#include <execution>
#include <iterator>

void test() {
  int a[] = {1};
  auto pred = [](auto) { return false; };
  std::all_of(std::execution::par, std::begin(a), std::end(a), pred);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::any_of(std::execution::par, std::begin(a), std::end(a), pred);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::none_of(std::execution::par, std::begin(a), std::end(a), pred);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::is_partitioned(std::execution::par, std::begin(a), std::end(a), pred); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
