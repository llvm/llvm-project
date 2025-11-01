//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<bool OtherConst>
//   requires sized_sentinel_for<zentinel<Const>, ziperator<OtherConst>>
// friend constexpr range_difference_t<maybe-const<OtherConst, InnerView>>
//   operator-(const sentinel& x, const iterator<OtherConst>& y);

#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>
#include <tuple>

#include "../types.h"
#include "../../range_adaptor_types.h"

template <class T, class U>
concept HasMinus = std::invocable<std::minus<>, const T&, const U&>;

template <class T>
concept SentinelHasMinus = HasMinus<std::ranges::sentinel_t<T>, std::ranges::iterator_t<T>>;

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[3] = {1, 2, 3};

  {
    // shortest range
    std::ranges::zip_transform_view v(MakeTuple{}, std::views::iota(0, 3), ForwardSizedNonCommon(buffer1));
    static_assert(!std::ranges::common_range<decltype(v)>);
    auto it = v.begin();
    auto st = v.end();
    assert(st - it == 3);
    assert(st - std::ranges::next(it, 1) == 2);

    assert(it - st == -3);
    assert(std::ranges::next(it, 1) - st == -2);
    static_assert(SentinelHasMinus<decltype(v)>);
  }

  {
    // underlying sentinel does not model sized_sentinel_for
    std::ranges::zip_transform_view v(MakeTuple{}, std::views::iota(0), SizedRandomAccessView(buffer1));
    static_assert(!std::ranges::common_range<decltype(v)>);
    static_assert(!SentinelHasMinus<decltype(v)>);
  }

  {
    // const incompatible:
    // underlying const sentinels cannot subtract underlying iterators
    // underlying sentinels cannot subtract underlying const iterators
    std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleForwardSizedNonCommon{buffer1});
    static_assert(!std::ranges::common_range<decltype(v)>);
    static_assert(!simple_view<decltype(v)>);

    using Iter      = std::ranges::iterator_t<decltype(v)>;
    using ConstIter = std::ranges::iterator_t<const decltype(v)>;
    static_assert(!std::is_same_v<Iter, ConstIter>);
    using Sentinel      = std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);
    auto it       = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = std::as_const(v).end();
    assert(it - st == -5);
    assert(st - it == 5);
    assert(const_it - const_st == -5);
    assert(const_st - const_it == 5);

    static_assert(!HasMinus<Iter, ConstSentinel>);
    static_assert(!HasMinus<ConstSentinel, Iter>);
    static_assert(!HasMinus<ConstIter, Sentinel>);
    static_assert(!HasMinus<Sentinel, ConstIter>);
  }

  {
    // const compatible allow non-const to const conversion
    std::ranges::zip_transform_view v(MakeTuple{}, ConstCompatibleForwardSized{buffer1});
    static_assert(!std::ranges::common_range<decltype(v)>);
    static_assert(!simple_view<decltype(v)>);

    using Iter      = std::ranges::iterator_t<decltype(v)>;
    using ConstIter = std::ranges::iterator_t<const decltype(v)>;
    static_assert(!std::is_same_v<Iter, ConstIter>);
    using Sentinel      = std::ranges::sentinel_t<decltype(v)>;
    using ConstSentinel = std::ranges::sentinel_t<const decltype(v)>;
    static_assert(!std::is_same_v<Sentinel, ConstSentinel>);

    static_assert(HasMinus<Iter, Sentinel>);
    static_assert(HasMinus<Sentinel, Iter>);
    static_assert(HasMinus<ConstIter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, ConstIter>);
    static_assert(HasMinus<Iter, ConstSentinel>);
    static_assert(HasMinus<ConstSentinel, Iter>);
    static_assert(HasMinus<ConstIter, Sentinel>);
    static_assert(HasMinus<Sentinel, ConstIter>);

    auto it       = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = std::as_const(v).end();

    assert(it - st == -5);
    assert(st - it == 5);
    assert(const_it - const_st == -5);
    assert(const_st - const_it == 5);
    assert(it - const_st == -5);
    assert(const_st - it == 5);
    assert(const_it - st == -5);
    assert(st - const_it == 5);
  }

  auto testMinus = [](auto&& v, auto distance) {
    auto it       = v.begin();
    auto const_it = std::as_const(v).begin();
    auto st       = v.end();
    auto const_st = std::as_const(v).end();

    assert(it - st == -distance);
    assert(st - it == distance);
    assert(const_it - const_st == -distance);
    assert(const_st - const_it == distance);
    assert(it - const_st == -distance);
    assert(const_st - it == distance);
    assert(const_it - st == -distance);
    assert(st - const_it == distance);
  };

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, ConstCompatibleForwardSized{buffer1});
    testMinus(v, 5);
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, ConstCompatibleForwardSized{buffer1}, std::views::iota(0, 100));
    testMinus(v, 5);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(
        Tie{},
        ConstCompatibleForwardSized{buffer1},
        ConstCompatibleForwardSized{buffer2},
        std::ranges::single_view(2.));
    testMinus(v, 1);
  }

  {
    // single empty range
    std::ranges::zip_transform_view v(MakeTuple{}, ConstCompatibleForwardSized(nullptr, 0));
    testMinus(v, 0);
  }

  {
    // empty range at the beginning
    std::ranges::zip_transform_view v(
        MakeTuple{}, std::ranges::empty_view<int>(), ConstCompatibleForwardSized{buffer1}, SimpleCommon{buffer2});
    testMinus(v, 0);
  }

  {
    // empty range in the middle
    std::ranges::zip_transform_view v(
        MakeTuple{},
        ConstCompatibleForwardSized{buffer1},
        std::ranges::empty_view<int>(),
        ConstCompatibleForwardSized{buffer2});
    testMinus(v, 0);
  }

  {
    // empty range at the end
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer1}, ConstCompatibleForwardSized{buffer2}, std::ranges::empty_view<int>());
    testMinus(v, 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
