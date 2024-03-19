//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// Test the libc++ extension that std::views::chunk is marked as [[nodiscard]].

#include <ranges>

void test() {
  int range[] = {1, 2, 3, 0, 1, 2};

  std::views::chunk(2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::chunk(
      range, 2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::chunk(
              2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::all |
      std::views::chunk(
          2); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
