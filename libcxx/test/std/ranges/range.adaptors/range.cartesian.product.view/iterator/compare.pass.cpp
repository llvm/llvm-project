//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// friend constexpr bool operator==(const iterator& x, const iterator& y)
//   requires equality_comparable<iterator_t<maybe-const<Const, First>>>;
// friend constexpr bool operator==(const iterator&, default_sentinel_t);
// friend constexpr auto operator<=>(const iterator& x, const iterator& y)
//   requires cartesian-product-all-random-access<...>;

#include <array>
#include <cassert>
#include <compare>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

#include "../../range_adaptor_types.h"

constexpr void compareOperatorTest(auto&& it1, auto&& it2) {
  assert(!(it1 < it1));
  assert(it1 < it2);
  assert(!(it2 < it1));
  assert(it1 <= it1);
  assert(it1 <= it2);
  assert(!(it2 <= it1));
  assert(!(it1 > it1));
  assert(!(it1 > it2));
  assert(it2 > it1);
  assert(it1 >= it1);
  assert(!(it1 >= it2));
  assert(it2 >= it1);
  assert(it1 == it1);
  assert(!(it1 == it2));
  assert(it2 == it2);
  assert(!(it1 != it1));
  assert(it1 != it2);
  assert(!(it2 != it2));
}

template <class I1, class I2>
constexpr void inequalityOperatorsDoNotExist(const I1&, const I2&) {
  static_assert(!std::is_invocable_v<std::less<>, I1, I2>);
  static_assert(!std::is_invocable_v<std::less_equal<>, I1, I2>);
  static_assert(!std::is_invocable_v<std::greater<>, I1, I2>);
  static_assert(!std::is_invocable_v<std::greater_equal<>, I1, I2>);
}

constexpr bool test() {
  { // <=> with three_way_comparable underlying iterators
    using It  = three_way_contiguous_iterator<int*>;
    using Sub = std::ranges::subrange<It>;
    static_assert(std::three_way_comparable<It>);

    int a[]  = {1, 2, 3, 4};
    int b[]  = {5, 6, 7};
    auto v   = std::views::cartesian_product(Sub(It(a), It(a + 4)), Sub(It(b), It(b + 3)));
    auto it1 = v.begin();
    auto it2 = it1 + 1;

    using Iter = decltype(it1);
    static_assert(std::three_way_comparable<Iter>);
    compareOperatorTest(it1, it2);
    assert((it1 <=> it2) == std::strong_ordering::less);
    assert((it1 <=> it1) == std::strong_ordering::equal);
    assert((it2 <=> it1) == std::strong_ordering::greater);
  }

  { // RA underlying iterators without <=> still get a synthesised three-way ordering
    using It  = random_access_iterator<int*>;
    using Sub = std::ranges::subrange<It>;
    static_assert(!std::three_way_comparable<It>);

    int a[]  = {1, 2, 3, 4};
    int b[]  = {5, 6, 7};
    auto v   = std::views::cartesian_product(Sub(It(a), It(a + 4)), Sub(It(b), It(b + 3)));
    auto it1 = v.begin();
    auto it2 = it1 + 1;

    using Iter = decltype(it1);
    static_assert(std::three_way_comparable<Iter>);
    compareOperatorTest(it1, it2);
  }

  { // input_range -- only == is available, no relational ops, no <=>
    std::array a{1};
    std::array b{10, 20};
    std::ranges::cartesian_product_view v(InputCommonView{a}, ForwardSizedView{b});
    using View = decltype(v);
    static_assert(!std::ranges::forward_range<View>);
    static_assert(std::ranges::input_range<View>);

    auto it = v.begin();
    auto en = v.end();
    assert(it != en);
    inequalityOperatorsDoNotExist(it, en);
  }

  { // forward+sized -- equality but no <=> (because not all-random-access)
    std::array a{1, 2};
    std::array b{3, 4};
    std::ranges::cartesian_product_view v(ForwardSizedView{a}, ForwardSizedView{b});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(std::equality_comparable<Iter>);
    static_assert(!std::three_way_comparable<Iter>);

    auto it1 = v.begin();
    auto it2 = it1;
    ++it2;
    assert(it1 != it2);
    inequalityOperatorsDoNotExist(it1, it2);
  }

  { // iterator vs default_sentinel_t -- at-end check via OR-fold over all positions
    std::array a{1, 2, 3};
    std::ranges::cartesian_product_view v(a);
    auto it = v.begin();
    assert(it != std::default_sentinel);
    ++it;
    ++it;
    assert(it != std::default_sentinel);
    ++it; // past last -> at end
    assert(it == std::default_sentinel);
  }

  { // sentinel is reached as soon as any range is at end (empty middle range)
    std::array a{1, 2};
    std::ranges::empty_view<int> empty;
    std::array b{3, 4};
    std::ranges::cartesian_product_view v(a, empty, b);
    auto it = v.begin();
    // The middle iterator is at end of empty range immediately, so begin == default_sentinel.
    assert(it == std::default_sentinel);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
