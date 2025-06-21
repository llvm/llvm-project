//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// template<class R>
//   enumerate_view(R&&) -> enumerate_view<views::all_t<R>>;

#include <cassert>
#include <ranges>

#include "test_iterators.h"

#include "types.h"

constexpr bool test() {
  {
    MinimalDefaultConstructedView jv;
    std::ranges::enumerate_view view(jv);
    static_assert(std::is_same_v<decltype(view), std::ranges::enumerate_view<MinimalDefaultConstructedView>>);
  }

  // Test with a range that isn't a view, to make sure we properly use views::all_t in the implementation.
  {
    NotAViewRange range;
    std::ranges::enumerate_view view(range);
    static_assert(std::is_same_v<decltype(view), std::ranges::enumerate_view<std::ranges::ref_view<NotAViewRange>>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
