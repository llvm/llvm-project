//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::cartesian_product_view

#include <cassert>
#include <ranges>

#include "test_iterators.h"

struct DefaultConstructibleView : std::ranges::view_base {
  int i_;
  int* begin();
  int* end();
};

constexpr bool test() {
  { // view | views::as_rvalue
    // DefaultConstructibleView v{{}, 3};
    [[maybe_unused]] std::ranges::cartesian_product_view<DefaultConstructibleView> r1;
    [[maybe_unused]] std::ranges::cartesian_product_view<DefaultConstructibleView, DefaultConstructibleView> r2;
  }
  return true;
}

int main() {
  test();
  static_assert(test());
}
