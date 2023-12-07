//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr decltype(auto) operator[](difference_type n) const
//   requires random_access_range<Base>

#include <cassert>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

template <class T, class U>
concept CanSubscript = requires(T t, U u) { t[u]; };

template <class BaseRange>
using ElemIter = std::ranges::iterator_t<std::ranges::elements_view<BaseRange, 0>>;

using RandomAccessRange = std::ranges::subrange<std::tuple<int>*>;
static_assert(std::ranges::random_access_range<RandomAccessRange>);

static_assert(CanSubscript<ElemIter<RandomAccessRange>, int>);

using BidiRange = std::ranges::subrange<bidirectional_iterator<std::tuple<int>*>>;
static_assert(!std::ranges::random_access_range<BidiRange>);

static_assert(!CanSubscript<ElemIter<BidiRange>, int>);

constexpr bool test() {
  {
    // reference
    std::tuple<int> ts[] = {{1}, {2}, {3}, {4}};
    auto ev              = ts | std::views::elements<0>;
    auto it              = ev.begin();

    assert(&it[0] == &*it);
    assert(&it[2] == &*(it + 2));

    static_assert(std::is_same_v<decltype(it[2]), int&>);
  }

  {
    // value
    auto ev = std::views::iota(0, 5) | std::views::transform([](int i) { return std::tuple<int>{i}; }) |
              std::views::elements<0>;
    auto it = ev.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(std::is_same_v<decltype(it[2]), int>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
