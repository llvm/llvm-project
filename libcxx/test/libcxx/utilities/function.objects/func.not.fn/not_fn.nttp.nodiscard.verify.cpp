//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// Test the libc++ extension that std::not_fn<NTTP> is marked as [[nodiscard]].

#include <functional>
#include <type_traits>

void test() {
  using F = std::true_type;
  std::not_fn<F{}>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
