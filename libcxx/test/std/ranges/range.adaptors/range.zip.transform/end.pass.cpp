//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto end()
// constexpr auto end() const
//   requires range<const InnerView> &&
//            regular_invocable<const F&, range_reference_t<const Views>...>;

#include <ranges>

#include "types.h"

template <class T>
concept HasConstEnd = requires(const T& ct) { ct.end(); };

template <class T>
concept HasEnd = requires(T& t) { t.end(); };

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // simple test
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, std::views::iota(0), std::ranges::single_view(2.));
    assert(v.begin() != v.end());
    assert(std::as_const(v).begin() != std::as_const(v).end());
    assert(v.begin() + 1 == v.end());
    assert(std::as_const(v).begin() + 1 == std::as_const(v).end());
  }

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it = v.begin();
    assert(it + 8 == v.end());
    assert(it + 8 == std::as_const(v).end());
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    auto it = v.begin();
    assert(it + 8 == v.end());
    assert(it + 8 == std::as_const(v).end());
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::single_view(2.));
    auto it = v.begin();
    assert(it + 1 == v.end());
    assert(it + 1 == std::as_const(v).end());
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
    // common_range<InnerView>
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it       = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = std::as_const(v).end();

    static_assert(!std::same_as<decltype(it), decltype(const_it)>);
    static_assert(!std::same_as<decltype(st), decltype(const_st)>);
    static_assert(std::same_as<decltype(it), decltype(st)>);
    static_assert(std::same_as<decltype(const_it), decltype(const_st)>);

    assert(it + 8 == st);
    assert(const_it + 8 == const_st);
  }
  {
    // !common_range<InnerView>
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleNonCommon{buffer});
    auto it       = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = std::as_const(v).end();

    static_assert(!std::same_as<decltype(it), decltype(const_it)>);
    static_assert(!std::same_as<decltype(st), decltype(const_st)>);
    static_assert(!std::same_as<decltype(it), decltype(st)>);
    static_assert(!std::same_as<decltype(const_it), decltype(const_st)>);

    assert(it + 8 == st);
    assert(const_it + 8 == const_st);
  }

  {
    // underlying const R is not a range
    using ZTV = std::ranges::zip_transform_view<MakeTuple, SimpleCommon, NoConstBeginView>;
    static_assert(HasEnd<ZTV>);
    static_assert(!HasConstEnd<ZTV>);
  }

  {
    // Fn cannot invoke on const range
    using ZTV = std::ranges::zip_transform_view<NonConstOnlyFn, ConstNonConstDifferentView>;
    static_assert(HasEnd<ZTV>);
    static_assert(!HasConstEnd<ZTV>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
