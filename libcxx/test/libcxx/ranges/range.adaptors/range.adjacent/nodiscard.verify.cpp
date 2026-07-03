//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Check that functions are marked [[nodiscard]]

// TODO: this test should eventually be expanded to cover functions beyond
// reserve_hint; at that time, it should be std-at-least-c++23

#include <ranges>
#include <utility>
#include <vector>

#include "test_macros.h"

void test() {
  std::vector<int> range;
  auto v = std::views::adjacent<2>(range);

#if TEST_STD_VER >= 26
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.reserve_hint();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).reserve_hint();
#endif
}
