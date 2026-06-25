//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<V>
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const V>

#include <cassert>
#include <ranges>

#include "types.h"

// There is no reserve_hint function on a stride view over a view that
// is *not* an approximately sized range
static_assert(!std::ranges::approximately_sized_range<BasicTestView<cpp17_input_iterator<int*>>>);
static_assert(
    !std::ranges::approximately_sized_range<std::ranges::stride_view<BasicTestView<cpp17_input_iterator<int*>>>>);

// There is a reserve_hint function on a stride view over a view that
// *is *an approximately sized range
static_assert(std::ranges::approximately_sized_range<
              BasicTestView<int*, sentinel_wrapper<int*>, /*IsSized=*/false, /*IsApproximatelySized=*/true>>);
static_assert(std::ranges::approximately_sized_range<std::ranges::stride_view<
                  BasicTestView<int*, sentinel_wrapper<int*>, /*IsSized=*/false, /*IsApproximatelySized=*/true>>>);

using Iota = std::ranges::iota_view<int, int>;
using ApproximatelySizedView =
    BasicTestView<std::ranges::iterator_t<Iota>,
                  std::ranges::sentinel_t<Iota>,
                  /*IsSized=*/false,
                  /*IsApproximatelySized=*/true>;

constexpr ApproximatelySizedView make_approximately_sized(Iota&& iota) {
  return ApproximatelySizedView(std::ranges::begin(iota), std::ranges::end(iota));
}

constexpr bool test() {
  {
    // Test with stride as exact multiple of number of elements in view strided over.
    auto view    = make_approximately_sized(std::views::iota(0, 12));
    auto strided = std::views::stride(view, 3);
    static_assert(std::ranges::approximately_sized_range<decltype(strided)>);
    assert(strided.reserve_hint() == 4);
  }

  {
    // Test with stride as inexact multiple of number of elements in view strided over.
    auto view = make_approximately_sized(std::views::iota(0, 22));
    static_assert(std::ranges::approximately_sized_range<decltype(view)>);
    auto strided = std::views::stride(view, 3);
    static_assert(std::ranges::approximately_sized_range<decltype(strided)>);
    assert(strided.size() == 8);
  }

  {
    // Empty range.
    auto view    = make_approximately_sized(std::views::iota(0, 0));
    auto strided = view | std::views::stride(3);
    assert(strided.reserve_hint() == 0);
  }

  {
    // Stride larger than range size.
    auto view    = make_approximately_sized(std::views::iota(0, 3));
    auto strided = view | std::views::stride(10);
    assert(strided.reserve_hint() == 1);
  }

  {
    // Stride equal to range size.
    auto view    = make_approximately_sized(std::views::iota(0, 3));
    auto strided = view | std::views::stride(5);
    assert(strided.reserve_hint() == 1);
  }

  {
    // Stride of 1.
    auto view    = make_approximately_sized(std::views::iota(0, 7));
    auto strided = view | std::views::stride(1);
    assert(strided.size() == 7);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
