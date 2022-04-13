//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template <class R, class P>
// lazy_split_view(R&&, P&&) -> lazy_split_view<views::all_t<R>, views::all_t<P>>;
//
// template <input_range R>
// lazy_split_view(R&&, range_value_t<R>) -> lazy_split_view<views::all_t<R>, single_view<range_value_t<R>>>;

#include <ranges>

#include <concepts>
#include <type_traits>
#include <utility>
#include "types.h"

struct ForwardRange {
  forward_iterator<const char*> begin() const;
  forward_iterator<const char*> end() const;
};
static_assert( std::ranges::forward_range<ForwardRange>);

struct InputRange {
  cpp20_input_iterator<const char*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<const char*>> end() const;
};
static_assert(std::ranges::input_range<InputRange>);

template <class I1, class I2, class ExpectedView, class ExpectedPattern>
constexpr void test() {
  I1 i1{};
  I2 i2{};

  std::ranges::lazy_split_view v(std::move(i1), std::move(i2));
  static_assert(std::same_as<decltype(v), std::ranges::lazy_split_view<ExpectedView, ExpectedPattern>>);
  using O = decltype(std::move(v).base());
  static_assert(std::same_as<O, ExpectedView>);
}

constexpr void testCtad() {
  // (Range, Pattern)
  test<ForwardView, ForwardView, ForwardView, ForwardView>();
  test<ForwardRange, ForwardRange, std::ranges::views::all_t<ForwardRange>, std::ranges::views::all_t<ForwardRange>>();

  // (Range, RangeElement)
  test<ForwardRange, char, std::ranges::views::all_t<ForwardRange>, std::ranges::single_view<char>>();
  test<InputRange, char, std::ranges::views::all_t<InputRange>, std::ranges::single_view<char>>();

  // (Range, RangeElement) with implicit conversion.
  test<ForwardRange, bool, std::ranges::views::all_t<ForwardRange>, std::ranges::single_view<char>>();
  test<InputRange, bool, std::ranges::views::all_t<InputRange>, std::ranges::single_view<char>>();

  // Note: CTAD from (InputRange, ForwardTinyRange) doesn't work -- the deduction guide wraps the pattern in
  // `views::all_t`, resulting in `views::owning_view<ForwardTinyRange>`. That type would never satisfy `tiny-range`
  // because `views::owning_view` contains a member function `size()` that shadows the static `size()` in
  // `ForwardTinyRange`.
}
