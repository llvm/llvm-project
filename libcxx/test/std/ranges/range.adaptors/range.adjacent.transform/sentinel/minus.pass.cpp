//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<bool OtherConst>
//   requires sized_sentinel_for<inner-sentinel<Const>, inner-iterator<OtherConst>>
// friend constexpr range_difference_t<maybe-const<OtherConst, InnerView>>
//   operator-(const iterator<OtherConst>& x, const sentinel& y);

// template<bool OtherConst>
//   requires sized_sentinel_for<inner-sentinel<Const>, inner-iterator<OtherConst>>
// friend constexpr range_difference_t<maybe-const<OtherConst, InnerView>>
//   operator-(const sentinel& x, const iterator<OtherConst>& y);

#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>
#include <tuple>
#include <utility>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

// clang-format off
template <class T, class U>
concept HasMinus = std::invocable<std::minus<>,const T&, const U&>;

template <class T>
concept SentinelHasMinus = HasMinus<std::ranges::sentinel_t<T>, std::ranges::iterator_t<T>>;
// clang-format on

template <std::size_t N, class Fn>
constexpr void test() {
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  {
    // underlying simple-view
    using View = std::ranges::adjacent_transform_view<ForwardSizedNonCommon, Fn, N>;
    static_assert(!std::ranges::common_range<View>);
    // but adjacent_transform_view is never a simple-view
    static_assert(!simple_view<View>);

    View v{ForwardSizedNonCommon(buffer), Fn{}};

    auto it = v.begin();
    auto st = v.end();
    assert(st - it == (10 - N));
    assert(st - std::ranges::next(it, 1) == (9 - N));

    assert(it - st == (static_cast<int>(N) - 10));
    assert(std::ranges::next(it, 1) - st == (static_cast<int>(N) - 9));
    static_assert(SentinelHasMinus<View>);
  }

  {
    // empty range
    using View = std::ranges::adjacent_transform_view<ForwardSizedNonCommon, Fn, N>;
    View v{ForwardSizedNonCommon(buffer, 0), Fn{}};

    auto it = v.begin();
    auto st = v.end();
    assert(st - it == 0);
    assert(it - st == 0);
  }

  {
    // N > size of underlying range
    using View = std::ranges::adjacent_transform_view<ForwardSizedNonCommon, Fn, 5>;
    View v{ForwardSizedNonCommon(buffer, 3), Fn{}};

    auto it = v.begin();
    auto st = v.end();
    assert(st - it == 0);
    assert(it - st == 0);
  }

  {
    // underlying sentinel does not model sized_sentinel_for
    using View = std::ranges::adjacent_transform_view<decltype(std::views::iota(0)), MakeTuple, N>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(!SentinelHasMinus<View>);
  }

  {
    // const incompatible:
    // underlying const sentinels cannot subtract underlying iterators
    // underlying sentinels cannot subtract underlying const iterators
    using View = std::ranges::adjacent_transform_view<NonSimpleForwardSizedNonCommon, Fn, N>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(!simple_view<View>);

    using Iter      = std::ranges::iterator_t<View>;
    using ConstIter = std::ranges::iterator_t<const View>;
    static_assert(!std::is_same_v<Iter, ConstIter>);
    using Sentinel      = std::ranges::sentinel_t<View>;
    using ConstSentinel = std::ranges::sentinel_t<const View>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);

    View v{NonSimpleForwardSizedNonCommon{buffer}, Fn{}};

    auto it       = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = std::as_const(v).end();

    int n = N;

    assert(it - st == (n - 10));
    assert(st - it == (10 - n));
    assert(const_it - const_st == (n - 10));
    assert(const_st - const_it == (10 - n));

    static_assert(!HasMinus<Iter, ConstSentinel>);
    static_assert(!HasMinus<ConstSentinel, Iter>);
    static_assert(!HasMinus<ConstIter, Sentinel>);
    static_assert(!HasMinus<Sentinel, ConstIter>);
  }

  {
    // const compatible allow non-const to const conversion
    using View = std::ranges::adjacent_transform_view<ConstCompatibleForwardSized, Fn, N>;
    static_assert(!std::ranges::common_range<View>);
    static_assert(!simple_view<View>);

    using Iter      = std::ranges::iterator_t<View>;
    using ConstIter = std::ranges::iterator_t<const View>;
    static_assert(!std::is_same_v<Iter, ConstIter>);
    using Sentinel      = std::ranges::sentinel_t<View>;
    using ConstSentinel = std::ranges::sentinel_t<const View>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);
    static_assert(HasMinus<Iter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, Iter>);
    static_assert(HasMinus<ConstIter, Sentinel>);
    static_assert(HasMinus<Sentinel, ConstIter>);

    View v{ConstCompatibleForwardSized{buffer}, Fn{}};

    auto it       = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = std::as_const(v).end();

    int n = N;

    assert(it - st == (n - 10));
    assert(st - it == (10 - n));
    assert(const_it - const_st == (n - 10));
    assert(const_st - const_it == (10 - n));
    assert(it - const_st == (n - 10));
    assert(const_st - it == (10 - n));
    assert(const_it - st == (n - 10));
    assert(st - const_it == (10 - n));
  }
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple>();
  test<N, Tie>();
  test<N, GetFirst>();
  test<N, Multiply>();
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
