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

// constexpr auto operator[](difference_type n) const
//   requires random_access_range<Base>;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

template <class T, class U>
concept HasSubscriptOperator = requires(T t, U u) { t[u]; };

template <class BaseRange>
using EnumerateIterator = std::ranges::iterator_t<std::ranges::enumerate_view<BaseRange>>;

using Subrange = std::ranges::subrange<int*>;
static_assert(HasSubscriptOperator<EnumerateIterator<Subrange>, int>);

using BidirectionalRange = std::ranges::subrange<bidirectional_iterator<int*>>;
static_assert(!HasSubscriptOperator<EnumerateIterator<BidirectionalRange>, int>);

constexpr bool test() {
  // Reference
  {
    std::array ts = {90, 1, 2, 84};
    auto view     = ts | std::views::enumerate;
    auto it       = view.begin();

    using DifferenceT = std::iter_difference_t<decltype(it)>;
    static_assert(std::is_same_v<decltype(it[2]), std::tuple<DifferenceT, int&>>);

    assert((it[0] == std::tuple<DifferenceT, int>(0, 90)));
    assert((it[1] == std::tuple<DifferenceT, int>(1, 1)));
    assert((it[2] == std::tuple<DifferenceT, int>(2, 2)));
    assert((it[3] == std::tuple<DifferenceT, int>(3, 84)));
  }

  // Value
  {
    auto view = std::views::iota(0, 4) | std::views::enumerate;
    auto it   = view.begin();

    using DifferenceT = std::iter_difference_t<decltype(it)>;
    static_assert(std::is_same_v<decltype(it[2]), std::tuple<DifferenceT, int>>);

    assert((it[0] == std::tuple<DifferenceT, int>(0, 0)));
    assert((it[1] == std::tuple<DifferenceT, int>(1, 1)));
    assert((it[2] == std::tuple<DifferenceT, int>(2, 2)));
    assert((it[3] == std::tuple<DifferenceT, int>(3, 3)));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
