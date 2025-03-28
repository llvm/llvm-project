//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// expected-no-diagnostics

// constexpr auto size()

#include <ranges>

#include "__ranges/stride_view.h"
#include "test_iterators.h"
#include "types.h"

// There is no size member function on a stride view over a view that
// is *not* a sized range
static_assert(!std::ranges::sized_range<BasicTestView<cpp17_input_iterator<int*>>>);
static_assert(!std::ranges::sized_range<std::ranges::stride_view<BasicTestView<cpp17_input_iterator<int*>>>>);

// There is a size member function on a stride view over a view that
// *is* a sized range
static_assert(std::ranges::sized_range<BasicTestView<int*, sentinel_wrapper<int*>, true>>);
static_assert(std::ranges::sized_range<std::ranges::stride_view<BasicTestView<int*, sentinel_wrapper<int*>, true>>>);

constexpr bool test() {
  {
    // Test with stride as exact multiple of number of elements in view strided over.
    constexpr auto iota_twelve = std::views::iota(0, 12);
    static_assert(std::ranges::sized_range<decltype(iota_twelve)>);
    constexpr auto stride_iota_twelve = std::views::stride(iota_twelve, 3);
    static_assert(std::ranges::sized_range<decltype(stride_iota_twelve)>);
    static_assert(4 == stride_iota_twelve.size(), "Striding by 3 through a 12 member list has size 4.");
  }

  {
    // Test with stride as inexact multiple of number of elements in view strided over.
    constexpr auto iota_twenty_two = std::views::iota(0, 22);
    static_assert(std::ranges::sized_range<decltype(iota_twenty_two)>);
    constexpr auto stride_iota_twenty_two = std::views::stride(iota_twenty_two, 3);
    static_assert(std::ranges::sized_range<decltype(stride_iota_twenty_two)>);
    static_assert(8 == stride_iota_twenty_two.size(), "Striding by 3 through a 22 member list has size 8.");
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
