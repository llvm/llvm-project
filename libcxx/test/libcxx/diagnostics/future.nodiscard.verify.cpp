//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// UNSUPPORTED: no-threads

// check that <future> functions are marked [[nodiscard]]

// clang-format off

#include <future>

void test() {
  std::async([]() {});                   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::async(std::launch::any, []() {}); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
