//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires equality_comparable<iterator_t<Base>>;
// friend constexpr bool operator<(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator>(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires random_access_range<Base> && three_way_comparable<iterator_t<Base>>;

#include <compare>
#include <functional>
#include <ranges>
#include <tuple>

#include "test_iterators.h"
#include "test_range.h"

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

constexpr void inequalityOperatorsDoNotExistTest(const auto& iter1, const auto& iter2) {
  using Iter1 = decltype(iter1);
  using Iter2 = decltype(iter2);
  static_assert(!std::is_invocable_v<std::less<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::less_equal<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater_equal<>, Iter1, Iter2>);
}

constexpr bool test() {
  std::tuple<int> ts[] = {{1}, {2}, {3}};

  {
    // Test a new-school iterator with operator<=>; the iterator should also have operator<=>.
    using It       = three_way_contiguous_iterator<std::tuple<int>*>;
    using Subrange = std::ranges::subrange<It>;
    static_assert(std::three_way_comparable<It>);
    using R = std::ranges::elements_view<Subrange, 0>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<R>>);

    auto ev    = Subrange{It{&ts[0]}, It{&ts[0] + 3}} | std::views::elements<0>;
    auto iter1 = ev.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);

    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);
  }

  {
    // Test an old-school iterator with no operator<=>; the elements view iterator shouldn't have
    // operator<=> either.
    using It       = random_access_iterator<std::tuple<int>*>;
    using Subrange = std::ranges::subrange<It>;
    static_assert(!std::three_way_comparable<It>);
    using R = std::ranges::elements_view<Subrange, 0>;
    static_assert(!std::three_way_comparable<std::ranges::iterator_t<R>>);

    auto ev    = Subrange{It{&ts[0]}, It{&ts[0] + 3}} | std::views::elements<0>;
    auto iter1 = ev.begin();
    auto iter2 = iter1 + 1;

    compareOperatorTest(iter1, iter2);
  }

  {
    // non random_access_range
    using It       = bidirectional_iterator<std::tuple<int>*>;
    using Subrange = std::ranges::subrange<It>;
    static_assert(!std::ranges::random_access_range<Subrange>);
    using R = std::ranges::elements_view<Subrange, 0>;
    static_assert(!std::three_way_comparable<std::ranges::iterator_t<R>>);

    auto ev = Subrange{It{&ts[0]}, It{&ts[0] + 1}} | std::views::elements<0>;

    auto it1 = ev.begin();
    auto it2 = ev.end();

    assert(it1 == it1);
    assert(!(it1 != it1));
    assert(it2 == it2);
    assert(!(it2 != it2));

    assert(it1 != it2);

    ++it1;
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // underlying iterator does not support ==
    using Iter     = cpp20_input_iterator<std::tuple<int>*>;
    using Sent     = sentinel_wrapper<Iter>;
    using Subrange = std::ranges::subrange<Iter, Sent>;
    using R        = std::ranges::elements_view<Subrange, 0>;
    static_assert(!std::three_way_comparable<std::ranges::iterator_t<R>>);

    auto ev = Subrange{Iter{&ts[0]}, Sent{Iter{&ts[0] + 1}}} | std::views::elements<0>;
    auto it = ev.begin();

    using ElemIter = decltype(it);
    static_assert(!weakly_equality_comparable_with<ElemIter, ElemIter>);
    inequalityOperatorsDoNotExistTest(it, it);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
