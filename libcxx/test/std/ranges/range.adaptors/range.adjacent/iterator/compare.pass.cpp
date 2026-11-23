//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr bool operator==(const iterator& x, const iterator& y);
// friend constexpr bool operator<(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator>(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator<=(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr bool operator>=(const iterator& x, const iterator& y)
//   requires random_access_range<Base>;
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires random_access_range<Base> &&
//            three_way_comparable<iterator_t<Base>>;

#include <cassert>
#include <ranges>
#include <compare>

#include "test_iterators.h"
#include "test_range.h"

#include "../../range_adaptor_types.h"

constexpr void compareOperatorTest(auto&& iter1, auto&& iter2) {
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

constexpr void inequalityOperatorsDoNotExistTest(auto&& iter1, auto&& iter2) {
  using Iter1 = decltype(iter1);
  using Iter2 = decltype(iter2);
  static_assert(!std::is_invocable_v<std::less<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::less_equal<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater<>, Iter1, Iter2>);
  static_assert(!std::is_invocable_v<std::greater_equal<>, Iter1, Iter2>);
}

template <std::size_t N>
constexpr void test() {
  {
    // Test a new-school iterator with operator<=>; the iterator should also have operator<=>.
    using It       = three_way_contiguous_iterator<int*>;
    using SubRange = std::ranges::subrange<It>;
    static_assert(std::three_way_comparable<It>);
    using R = std::ranges::adjacent_view<SubRange, N>;
    static_assert(std::three_way_comparable<std::ranges::iterator_t<R>>);

    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto r       = R{SubRange(It(buffer), It(buffer + 8))};
    auto iter1   = r.begin();
    auto iter2   = iter1 + 1;

    compareOperatorTest(iter1, iter2);

    assert((iter1 <=> iter2) == std::strong_ordering::less);
    assert((iter1 <=> iter1) == std::strong_ordering::equal);
    assert((iter2 <=> iter1) == std::strong_ordering::greater);
  }

  {
    // Test an old-school iterator with no operator<=>; the adjacent iterator shouldn't have
    // operator<=> either.
    using It       = random_access_iterator<int*>;
    using SubRange = std::ranges::subrange<It>;
    static_assert(!std::three_way_comparable<It>);
    using R = std::ranges::adjacent_view<SubRange, N>;
    static_assert(!std::three_way_comparable<std::ranges::iterator_t<R>>);

    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto r       = R{SubRange(It(buffer), It(buffer + 8))};
    auto iter1   = r.begin();
    auto iter2   = iter1 + 1;

    compareOperatorTest(iter1, iter2);
  }

  {
    // non random_access_range
    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

    std::ranges::adjacent_view<BidiCommonView, N> v(BidiCommonView{buffer});
    using View = decltype(v);
    static_assert(!std::ranges::random_access_range<View>);

    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 != it2);

    // advance it1 to the end
    std::ranges::advance(it1, 9 - N);
    assert(it1 == it2);

    inequalityOperatorsDoNotExistTest(it1, it2);
  }

  {
    // empty range
    auto v   = std::views::empty<int> | std::views::adjacent<N>;
    auto it1 = v.begin();
    auto it2 = v.end();
    assert(it1 == it2);
  }

  {
    // N > size of range
    int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto v       = std::ranges::adjacent_view<ContiguousCommonView, 10>(ContiguousCommonView{buffer});
    auto it1     = v.begin();
    auto it2     = v.end();
    assert(it1 == it2);
  }
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
