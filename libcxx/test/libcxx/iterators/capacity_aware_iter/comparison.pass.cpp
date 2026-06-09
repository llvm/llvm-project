//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Iterator, class _Container, class _ContainerMaxElements>
// struct __capacity_aware_iterator;

// Comparison operators

// operator==
// operator!=
// operator<
// operator<=
// operator>
// operator>=
// operator<=>

#include <__iterator/capacity_aware_iterator.h>
#include <cassert>
#include <compare>
#include <concepts>
#include <iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <typename Iter>
constexpr bool test() {
  int arr[]         = {1, 2, 3, 4};
  constexpr long sz = std::size(arr);

  using CapIter = std::__capacity_aware_iterator<Iter, decltype(arr), sz>;

  CapIter iter1 = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(arr));
  CapIter iter2 = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(arr + 4));

  // operator==
  {
    std::same_as<bool> decltype(auto) res = iter1 == iter1;
    ASSERT_NOEXCEPT(iter1 == iter2);
    assert(res);
    assert(!(iter1 == iter2));
  }

  // operator!=
  {
    std::same_as<bool> decltype(auto) res = iter1 != iter2;
    ASSERT_NOEXCEPT(iter1 != iter2);
    assert(res);
    assert(!(iter1 != iter1));
  }

  // operator<
  {
    std::same_as<bool> decltype(auto) res = iter1 < iter2;
    ASSERT_NOEXCEPT(iter1 < iter2);
    assert(res);
    assert(!(iter1 < iter1));
    assert(!(iter2 < iter1));
  }

  // operator<=
  {
    std::same_as<bool> decltype(auto) res = iter1 <= iter2;
    ASSERT_NOEXCEPT(iter1 <= iter2);
    assert(res);
    assert(iter1 <= iter1);
    assert(!(iter2 <= iter1));
  }

  // operator>
  {
    std::same_as<bool> decltype(auto) res = iter1 > iter2;
    ASSERT_NOEXCEPT(iter1 > iter2);
    assert(iter2 > iter1);
    assert(!res);
    assert(!(iter1 > iter1));
  }

  // operator>=
  {
    std::same_as<bool> decltype(auto) res = iter1 >= iter2;
    ASSERT_NOEXCEPT(iter1 >= iter2);
    assert(iter2 >= iter1);
    assert(iter1 >= iter1);
    assert(!res);
  }

  // operator <=>
  {
    ASSERT_NOEXCEPT(iter1 <=> iter2);
    std::same_as<std::strong_ordering> decltype(auto) r1 = iter1 <=> iter2;
    assert(r1 == std::strong_ordering::less);

    std::same_as<std::strong_ordering> decltype(auto) r2 = iter2 <=> iter1;
    assert(r2 == std::strong_ordering::greater);

    std::same_as<std::strong_ordering> decltype(auto) r3 = iter1 <=> iter1;
    assert(r3 == std::strong_ordering::equal);
    assert(r3 == std::strong_ordering::equivalent);
  }

  return true;
}

int main(int, char**) {
  test<contiguous_iterator<int*>>();
  static_assert(test<contiguous_iterator<int*>>());

  return 0;
}
