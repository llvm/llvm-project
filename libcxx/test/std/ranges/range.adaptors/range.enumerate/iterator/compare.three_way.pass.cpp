//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// friend constexpr strong_ordering operator<=>(const iterator& x, const iterator& y) noexcept;

#include <cassert>
#include <ranges>

#include "test_iterators.h"

#include "../types.h"

constexpr void compareOperatorTest(const auto& iter1, const auto& iter2) {
  assert(!(iter1 < iter1));
  assert(iter1 < iter2);
  assert(!(iter2 < iter1));

  assert(iter1 <= iter1);
  assert(iter1 <= iter2);
  assert(!(iter2 <= iter1));

  assert(!(iter1 > iter1));
  assert(!(iter1 > iter2));
  assert(iter2 > iter1);

  assert(iter1 >= iter1);
  assert(!(iter1 >= iter2));
  assert(iter2 >= iter1);

  assert(iter1 == iter1);
  assert(!(iter1 == iter2));
  assert(iter2 == iter2);

  assert(!(iter1 != iter1));
  assert(iter1 != iter2);
  assert(!(iter2 != iter2));
}

constexpr bool test() {
  int buff[] = {0, 1, 2, 3};
  {
    using View = std::ranges::enumerate_view<RangeView>;

    using Iterator = std::ranges::iterator_t<View>;
    static_assert(std::three_way_comparable<Iterator>);
    using Subrange = std::ranges::subrange<Iterator>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<Subrange>>);
    using EnumerateView = std::ranges::enumerate_view<Subrange>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<EnumerateView>>);

    RangeView const range(buff, buff + 4);

    std::same_as<View> decltype(auto) ev = std::views::enumerate(range);

    auto it1 = ev.begin();
    auto it2 = it1 + 1;

    compareOperatorTest(it1, it2);

    assert((it1 <=> it2) == std::strong_ordering::less);
    assert((it1 <=> it1) == std::strong_ordering::equal);
    assert((it2 <=> it2) == std::strong_ordering::equal);
    assert((it2 <=> it1) == std::strong_ordering::greater);
  }

  // Test an old-school iterator with no operator<=>
  {
    using Iterator = random_access_iterator<int*>;
    static_assert(!std::three_way_comparable<Iterator>);
    using Subrange = std::ranges::subrange<Iterator>;
    static_assert(!std::three_way_comparable<std::ranges::iterator_t<Subrange>>);
    using EnumerateView = std::ranges::enumerate_view<Subrange>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<EnumerateView>>);

    auto ev  = Subrange{Iterator{buff}, Iterator{buff + 3}} | std::views::enumerate;
    auto it1 = ev.begin();
    auto it2 = it1 + 1;

    compareOperatorTest(it1, it2);

    assert((it1 <=> it2) == std::strong_ordering::less);
    assert((it1 <=> it1) == std::strong_ordering::equal);
    assert((it2 <=> it2) == std::strong_ordering::equal);
    assert((it2 <=> it1) == std::strong_ordering::greater);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
