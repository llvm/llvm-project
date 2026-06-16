//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Check that functions are marked [[nodiscard]]

#include <generator>

std::generator<int> generate_ints() {
  co_yield 1;
  co_yield 2;
  co_yield 3;
}

void test() {
  auto gen = generate_ints();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  gen.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  gen.end();
}
