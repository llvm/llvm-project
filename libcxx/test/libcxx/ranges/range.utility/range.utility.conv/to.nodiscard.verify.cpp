//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test the libc++ extension that `std::ranges::to` is marked as [[nodiscard]].

#include <ranges>
#include <vector>

void test() {
  using R = std::vector<int>;
  R in = {1, 2, 3};
  std::allocator<int> alloc;

  std::ranges::to<R>(in); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<R>(in, alloc); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector>(in); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::to<std::vector>(in, alloc); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  in | std::ranges::to<R>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  in | std::ranges::to<R>(alloc); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  in | std::ranges::to<std::vector>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  in | std::ranges::to<std::vector>(alloc); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
