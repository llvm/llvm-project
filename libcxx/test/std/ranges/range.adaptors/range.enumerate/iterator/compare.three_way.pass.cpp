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

template <typename Iter>
constexpr void testBase_with_CommonRange() {
  using CommonRange = MinimalRange<Iter, Iter>;
  static_assert(std::ranges::range<CommonRange>);
  static_assert(std::ranges::common_range<CommonRange>);

  int arr[] = {94, 1, 2, 82};

  CommonRange range{Iter{arr}, Iter{arr + 4}};

  auto ev = range | std::views::enumerate;

  const auto it1 = ev.begin();
  assert(*it1 == std::make_tuple(0, 94));
  const auto it2 = ++ev.begin();
  assert(*it2 == std::make_tuple(1, 1));

  std::same_as<std::strong_ordering> decltype(auto) result = it1 <=> it2;
  static_assert(noexcept(operator<=>(it1, it2)));

  assert(result == std::strong_ordering::less);
  assert((it1 <=> it1) == std::strong_ordering::equal);
  assert((it2 <=> it2) == std::strong_ordering::equal);
  assert((it2 <=> it1) == std::strong_ordering::greater);

  compareOperatorTest(it1, it2);
}

template <typename Iter>
constexpr void testBase_with_NonCommonRange() {
  using Sent           = sentinel_wrapper<Iter>;
  using NonCommonRange = MinimalRange<Iter, Sent>;
  static_assert(!std::ranges::common_range<NonCommonRange>);

  int arr[] = {94, 1, 2, 82};

  NonCommonRange range{Iter{arr}, Sent{Iter{arr + 4}}};

  auto ev = range | std::views::enumerate;

  const auto it1 = ev.begin();
  assert(*it1 == std::make_tuple(0, 94));
  const auto it2 = ++ev.begin();
  assert(*it2 == std::make_tuple(1, 1));

  std::same_as<std::strong_ordering> decltype(auto) result = it1 <=> it2;
  static_assert(noexcept(operator<=>(it1, it2)));

  assert(result == std::strong_ordering::less);
  assert((it1 <=> it1) == std::strong_ordering::equal);
  assert((it2 <=> it2) == std::strong_ordering::equal);
  assert((it2 <=> it1) == std::strong_ordering::greater);

  compareOperatorTest(it1, it2);
}

constexpr bool test() {
  testBase_with_CommonRange<forward_iterator<int*>>();
  testBase_with_CommonRange<bidirectional_iterator<int*>>();
  testBase_with_CommonRange<random_access_iterator<int*>>();
  testBase_with_CommonRange<contiguous_iterator<int*>>();
  testBase_with_CommonRange<int*>();
  testBase_with_CommonRange<int const*>();

  testBase_with_NonCommonRange<cpp17_input_iterator<int*>>();
  testBase_with_NonCommonRange<forward_iterator<int*>>();
  testBase_with_NonCommonRange<bidirectional_iterator<int*>>();
  testBase_with_NonCommonRange<random_access_iterator<int*>>();
  testBase_with_NonCommonRange<contiguous_iterator<int*>>();
  testBase_with_NonCommonRange<int*>();
  testBase_with_NonCommonRange<int const*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
