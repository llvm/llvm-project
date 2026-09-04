//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <span>

// constexpr bool empty() const noexcept;

#include <span>

void test() {
  std::span<int> s1;
  s1.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int arr[] = {1, 2, 3};
  std::span<int, 3> s2{arr};
  s2.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
