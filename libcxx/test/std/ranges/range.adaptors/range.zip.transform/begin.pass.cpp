//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin();
// constexpr auto begin() const
//   requires range<const InnerView> &&
//            regular_invocable<const F&, range_reference_t<const Views>...>;

#include <ranges>

#include <cassert>
#include <concepts>

#include "types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // all underlying iterators should be at the begin position
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, std::views::iota(0), std::ranges::single_view(2.));
    auto it = v.begin();
    assert(*it == std::make_tuple(1, 0, 2.0));

    auto const_it = std::as_const(v).begin();
    assert(*const_it == *it);

    static_assert(!std::same_as<decltype(it), decltype(const_it)>);
  }

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it = v.begin();
    assert(*it == std::make_tuple(1));
    auto cit = std::as_const(v).begin();
    assert(*cit == std::make_tuple(1));
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    auto it = v.begin();
    assert(&*it == &buffer[0]);
    auto cit = std::as_const(v).begin();
    assert(&*cit == &buffer[0]);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::single_view(2.));
    auto it = v.begin();
    assert(&std::get<0>(*it) == &buffer[0]);
    assert(&std::get<1>(*it) == &buffer[0]);
    assert(std::get<2>(*it) == 2.0);
    auto cit = std::as_const(v).begin();
    assert(&std::get<0>(*cit) == &buffer[0]);
    assert(&std::get<1>(*cit) == &buffer[0]);
    assert(std::get<2>(*cit) == 2.0);
  }

  {
    // single empty range
    std::ranges::zip_transform_view v(MakeTuple{}, std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(std::as_const(v).begin() == std::as_const(v).end());
  }

  {
    // empty range at the beginning
    std::ranges::zip_transform_view v(
        MakeTuple{}, std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(v.begin() == v.end());
    assert(std::as_const(v).begin() == std::as_const(v).end());
  }

  {
    // empty range in the middle
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(v.begin() == v.end());
    assert(std::as_const(v).begin() == std::as_const(v).end());
  }

  {
    // empty range at the end
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(std::as_const(v).begin() == std::as_const(v).end());
  }

  {
    // underlying const R is not a range
    using ZTV = std::ranges::zip_transform_view<MakeTuple, SimpleCommon, NoConstBeginView>;
    static_assert(HasBegin<ZTV>);
    static_assert(!HasConstBegin<ZTV>);
  }

  {
    // Fn cannot be invoked on const range
    using ZTV = std::ranges::zip_transform_view<NonConstOnlyFn, ConstNonConstDifferentView>;
    static_assert(HasBegin<ZTV>);
    static_assert(!HasConstBegin<ZTV>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
