//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// class inplace_vector

// bool empty() const noexcept;

// See also: https://wg21.link/P2422R0
// In practice, this should be marked [[no_discard]]

#include <inplace_vector>

void f() {
  {
    std::inplace_vector<int, 10> c;
    c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::inplace_vector<int, 0> c;
    c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
