//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class R, class P>
// split_view(R&&, P&&) -> split_view<views::all_t<R>, views::all_t<P>>;
//
// template <input_range R>
// split_view(R&&, range_value_t<R>) -> split_view<views::all_t<R>, single_view<range_value_t<R>>>;

#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

struct Container {
  int* begin() const;
  int* end() const;
};

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

template <class I1, class I2, class ExpectedView, class ExpectedPattern>
constexpr void test() {
  static_assert(std::is_same_v<decltype(std::ranges::split_view(std::declval<I1>(), std::declval<I2>())),
                               std::ranges::split_view<ExpectedView, ExpectedPattern>>);
}

constexpr void testCtad() {
  // (Range, Pattern)
  test<View, View, View, View>();
  test<Container&, Container&, std::ranges::ref_view<Container>, std::ranges::ref_view<Container>>();
  test<Container&&, Container&&, std::ranges::owning_view<Container>, std::ranges::owning_view<Container>>();

  // (Range, RangeElement)
  test<Container&, int, std::ranges::ref_view<Container>, std::ranges::single_view<int>>();
  test<View, int, View, std::ranges::single_view<int>>();

  // (Range, RangeElement) with implicit conversion.
  test<Container&, bool, std::ranges::ref_view<Container>, std::ranges::single_view<int>>();
  test<View, bool, View, std::ranges::single_view<int>>();
}
