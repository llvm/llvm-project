//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <array> functions are marked [[nodiscard]]

#include <array>

void array_test() {
  std::array<int, 1> array;
  array.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void empty_array_test() {
  std::array<int, 0> array;
  array.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
