//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <string>
#include <utility>

void test() {
  // [range.lazy.split.overview]

  std::string str { "the quick brown fox" };
  char pattern    = ' ';

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::lazy_split(str, pattern);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::lazy_split(pattern);

}
