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

// class enumerate_view::iterator

// constexpr iterator& operator+=(difference_type x)
//   requires random_access_range<Base>;
// constexpr iterator& operator-=(difference_type x)
//   requires random_access_range<Base>;

//    friend constexpr iterator operator+(const iterator& x, difference_type y)
//      requires random_access_range<Base>;
//    friend constexpr iterator operator+(difference_type x, const iterator& y)
//      requires random_access_range<Base>;
//    friend constexpr iterator operator-(const iterator& x, difference_type y)
//      requires random_access_range<Base>;
//    friend constexpr difference_type operator-(const iterator& x, const iterator& y) noexcept;

#include <concepts>
#include <ranges>

#include "test_iterators.h"

// Concepts

template <class T, class U>
concept CanPlus = requires(T t, U u) { t + u; };

template <class T, class U>
concept CanPlusEqual = requires(T t, U u) { t += u; };

template <class T, class U>
concept CanMinus = requires(T t, U u) { t - u; };

template <class T, class U>
concept CanMinusEqual = requires(T t, U u) { t -= u; };

template <class BaseRange>
using EnumerateIter = std::ranges::iterator_t<std::ranges::enumerate_view<BaseRange>>;

using RandomAccessRange = std::ranges::subrange<int*>;

// SFINAE.

static_assert(std::ranges::random_access_range<RandomAccessRange>);
static_assert(
    std::sized_sentinel_for<std::ranges::iterator_t<RandomAccessRange>, std::ranges::iterator_t<RandomAccessRange>>);

static_assert(CanPlus<EnumerateIter<RandomAccessRange>, int>);
static_assert(CanPlus<int, EnumerateIter<RandomAccessRange>>);
static_assert(CanPlusEqual<EnumerateIter<RandomAccessRange>, int>);
static_assert(CanMinus<EnumerateIter<RandomAccessRange>, int>);
static_assert(CanMinus<EnumerateIter<RandomAccessRange>, EnumerateIter<RandomAccessRange>>);
static_assert(CanMinusEqual<EnumerateIter<RandomAccessRange>, int>);

using BidirectionalRange = std::ranges::subrange<bidirectional_iterator<int*>>;
static_assert(!std::ranges::random_access_range<BidirectionalRange>);
static_assert(
    !std::sized_sentinel_for<std::ranges::iterator_t<BidirectionalRange>, std::ranges::iterator_t<BidirectionalRange>>);

static_assert(!CanPlus<EnumerateIter<BidirectionalRange>, int>);
static_assert(!CanPlus<int, EnumerateIter<BidirectionalRange>>);
static_assert(!CanPlusEqual<EnumerateIter<BidirectionalRange>, int>);
static_assert(!CanMinus<EnumerateIter<BidirectionalRange>, int>);
static_assert(!CanMinusEqual<EnumerateIter<BidirectionalRange>, int>);

constexpr void test_with_common_range() {
  int ts[] = {90, 1, 2, 84};

  RandomAccessRange r{ts, ts + 4};
  auto ev = r | std::views::enumerate;

  using DifferenceT = std::ranges::range_difference_t<decltype(ev)>;

  // operator+(x, n), operator+(n,x) and operator+=
  {
    auto it1 = ev.begin();

    auto it2 = it1 + 3;
    assert(it2.base() == &ts[3]);

    auto it3 = 3 + it1;
    assert(it3.base() == &ts[3]);

    it1 += 3;
    assert(it1 == it2);
    assert(it1.base() == &ts[3]);
  }

  // operator-(x, n) and operator-=
  {
    auto it1 = ev.end();

    auto it2 = it1 - 3;
    assert(it2.base() == &ts[1]);

    it1 -= 3;
    assert(it1 == it2);
    assert(it1.base() == &ts[1]);
  }

  // friend constexpr difference_type operator-(const iterator& x, const iterator& y) noexcept;
  {
    auto it1 = ev.begin();
    auto it2 = ev.end();

    std::same_as<DifferenceT> decltype(auto) result = (it2 - it1);

    assert(result == 4);

    static_assert(noexcept(it1 - it2));

    assert(((it2 - 1) - (it1 + 1)) == 2);
  }
}

constexpr void test_with_noncommon_range() {
  int ts[] = {90, 1, 2, 84};

  RandomAccessRange r{ts, ts + 4};
  auto it = std::counted_iterator{r.begin(), std::ssize(r)};
  auto sr = std::ranges::subrange{it, std::default_sentinel};
  auto ev = sr | std::views::enumerate;

  using DifferenceT = std::ranges::range_difference_t<decltype(ev)>;

  // operator+(x, n), operator+(n,x) and operator+=
  {
    auto it1 = ev.begin();

    auto it2 = it1 + 3;
    assert(*it2.base() == 84);

    auto it3 = 3 + it1;
    assert(*it3.base() == 84);

    it1 += 3;
    assert(it1 == it2);
    assert(*it1.base() == 84);
  }

  // operator-(x, n) and operator-=
  {
    auto it1 = ev.begin();

    auto it2 = it1 + 3;
    assert(*(it2 - 1).base() == 2);

    it2 -= 3;
    assert(it1 == it2);
    assert(*it2.base() == 90);
  }

  // When the range is non-common, the ev.end() is a sentinel type.
}

constexpr bool test() {
  test_with_common_range();
  test_with_noncommon_range();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
