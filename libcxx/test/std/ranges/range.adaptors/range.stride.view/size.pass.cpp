//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "types.h"

// There is no size member function on a stride view over a view that
// is *not* a sized range
static_assert(!std::ranges::sized_range<BasicTestView<cpp17_input_iterator<int*>>>);
static_assert(!std::ranges::sized_range<std::ranges::stride_view<BasicTestView<cpp17_input_iterator<int*>>>>);

// There is a size member function on a stride view over a view that
// *is* a sized range
static_assert(std::ranges::sized_range<BasicTestView<int*, sentinel_wrapper<int*>, /*IsSized=*/true>>);
static_assert(
    std::ranges::sized_range<std::ranges::stride_view<BasicTestView<int*, sentinel_wrapper<int*>, /*IsSized=*/true>>>);

constexpr bool test() {
  {
    // Test with stride as exact multiple of number of elements in view strided over.
    auto iota = std::views::iota(0, 12);
    static_assert(std::ranges::sized_range<decltype(iota)>);
    auto strided = std::views::stride(iota, 3);
    static_assert(std::ranges::sized_range<decltype(strided)>);
    assert(strided.size() == 4);
  }

  {
    // Test with stride as inexact multiple of number of elements in view strided over.
    auto iota = std::views::iota(0, 22);
    static_assert(std::ranges::sized_range<decltype(iota)>);
    auto strided = std::views::stride(iota, 3);
    static_assert(std::ranges::sized_range<decltype(strided)>);
    assert(strided.size() == 8);
  }
  {
    // Empty range.
    auto strided = std::views::iota(0, 0) | std::views::stride(3);
    assert(strided.size() == 0);
  }
  {
    // Stride larger than range size.
    auto strided = std::views::iota(0, 3) | std::views::stride(10);
    assert(strided.size() == 1);
  }
  {
    // Stride equal to range size.
    auto strided = std::views::iota(0, 5) | std::views::stride(5);
    assert(strided.size() == 1);
  }
  {
    // Stride of 1.
    auto strided = std::views::iota(0, 7) | std::views::stride(1);
    assert(strided.size() == 7);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
