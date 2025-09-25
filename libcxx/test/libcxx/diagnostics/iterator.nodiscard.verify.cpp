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

#include "test_macros.h"

void test() {
  std::vector<int> container;
  int c_array[] = {1, 2, 3};
  std::initializer_list<int> initializer_list;

  std::empty(container);                                     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::empty(c_array);                                       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::empty(initializer_list);                              // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::prev(c_array);                                        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::next(c_array);                                        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 20
  std::ranges::prev(c_array);                                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::prev(container.end(), 2);                     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::next(container.end(), 2, container.begin());  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::next(c_array);                                // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::next(container.begin(), 2);                   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::next(container.end(), 1, container.end());    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
