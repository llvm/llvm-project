//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator& operator+=(difference_type n)
//    requires random_access_range<Base>;
//
// constexpr iterator& operator-=(difference_type n)
//   requires random_access_range<Base>;
//
// friend constexpr iterator operator+(const iterator& x, difference_type y)
//     requires random_access_range<Base>;
//
// friend constexpr iterator operator+(difference_type x, const iterator& y)
//   requires random_access_range<Base>;
//
// friend constexpr iterator operator-(const iterator& x, difference_type y)
//   requires random_access_range<Base>;
//
// friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//   requires sized_sentinel_for<iterator_t<Base>, iterator_t<Base>>;

#include <ranges>

#include <tuple>

#include "test_iterators.h"

template <class T, class U>
concept CanPlus = requires(T t, U u) { t + u; };

template <class T, class U>
concept CanPlusEqual = requires(T t, U u) { t += u; };

template <class T, class U>
concept CanMinus = requires(T t, U u) { t - u; };

template <class T, class U>
concept CanMinusEqual = requires(T t, U u) { t -= u; };

template <class BaseRange>
using ElemIter = std::ranges::iterator_t<std::ranges::elements_view<BaseRange, 0>>;

using RandomAccessRange = std::ranges::subrange<std::tuple<int>*>;
static_assert(std::ranges::random_access_range<RandomAccessRange>);
static_assert(std::sized_sentinel_for<std::ranges::iterator_t<RandomAccessRange>, //
                                      std::ranges::iterator_t<RandomAccessRange>>);

static_assert(CanPlus<ElemIter<RandomAccessRange>, int>);
static_assert(CanPlus<int, ElemIter<RandomAccessRange>>);
static_assert(CanPlusEqual<ElemIter<RandomAccessRange>, int>);
static_assert(CanMinus<ElemIter<RandomAccessRange>, int>);
static_assert(CanMinus<ElemIter<RandomAccessRange>, ElemIter<RandomAccessRange>>);
static_assert(CanMinusEqual<ElemIter<RandomAccessRange>, int>);

using BidiRange = std::ranges::subrange<bidirectional_iterator<std::tuple<int>*>>;
static_assert(!std::ranges::random_access_range<BidiRange>);
static_assert(!std::sized_sentinel_for<std::ranges::iterator_t<BidiRange>, //
                                       std::ranges::iterator_t<BidiRange>>);

static_assert(!CanPlus<ElemIter<BidiRange>, int>);
static_assert(!CanPlus<int, ElemIter<BidiRange>>);
static_assert(!CanPlusEqual<ElemIter<BidiRange>, int>);
static_assert(!CanMinus<ElemIter<BidiRange>, int>);
static_assert(!CanMinus<ElemIter<BidiRange>, ElemIter<BidiRange>>);
static_assert(!CanMinusEqual<ElemIter<BidiRange>, int>);

constexpr bool test() {
  std::tuple<int> ts[] = {{1}, {2}, {3}, {4}};

  RandomAccessRange r{&ts[0], &ts[0] + 4};
  auto ev = r | std::views::elements<0>;
  {
    // operator+(x, n) operator+(n,x) and operator+=
    auto it1 = ev.begin();

    auto it2 = it1 + 3;
    assert(it2.base() == &ts[3]);

    auto it3 = 3 + it1;
    assert(it3.base() == &ts[3]);

    it1 += 3;
    assert(it1 == it2);
    assert(it1.base() == &ts[3]);
  }

  {
    // operator-(x, n) and operator-=
    auto it1 = ev.end();

    auto it2 = it1 - 3;
    assert(it2.base() == &ts[1]);

    it1 -= 3;
    assert(it1 == it2);
    assert(it1.base() == &ts[1]);
  }

  {
    // operator-(x, y)
    assert((ev.end() - ev.begin()) == 4);

    auto it1 = ev.begin() + 2;
    auto it2 = ev.end() - 1;
    assert((it1 - it2) == -1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
