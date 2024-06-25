//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <utility> functions are marked [[nodiscard]]

// clang-format off

#include <utility>

#include "test_macros.h"

void test() {
  int i = 0;

  std::forward<int>(i);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::forward<int>(1);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(i);             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move_if_noexcept(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 17
  std::as_const(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 23
  enum E { Apple, Orange } e = Apple;
  std::to_underlying(e); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
