//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// check that <iterator> functions are marked [[nodiscard]]

// clang-format off

#include <iterator>
#include <vector>

void test() {
  std::vector<int> container;
  int c_array[] = {1, 2, 3};
  std::initializer_list<int> initializer_list;

  std::empty(container);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::empty(c_array);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::empty(initializer_list); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
