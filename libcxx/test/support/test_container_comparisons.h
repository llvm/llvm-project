//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//  Utility functions to test comparisons on containers.

#ifndef TEST_CONTAINER_COMPARISONS
#define TEST_CONTAINER_COMPARISONS

#include "test_comparisons.h"

// Implementation detail of `test_ordered_container_spaceship`
template <template <typename...> typename Container, typename Elem, typename Order>
constexpr void test_ordered_container_spaceship_with_type() {
  // Empty containers
  {
    Container<Elem> l1;
    Container<Elem> l2;
    assert(testOrder(l1, l2, Order::equivalent));
  }
  // Identical contents
  {
    Container<Elem> l1{1, 1};
    Container<Elem> l2{1, 1};
    assert(testOrder(l1, l2, Order::equivalent));
  }
  // Less, due to contained values
  {
    Container<Elem> l1{1, 1};
    Container<Elem> l2{1, 2};
    assert(testOrder(l1, l2, Order::less));
  }
  // Greater, due to contained values
  {
    Container<Elem> l1{1, 3};
    Container<Elem> l2{1, 2};
    assert(testOrder(l1, l2, Order::greater));
  }
  // Shorter list
  {
    Container<Elem> l1{1};
    Container<Elem> l2{1, 2};
    assert(testOrder(l1, l2, Order::less));
  }
  // Longer list
  {
    Container<Elem> l1{1, 2};
    Container<Elem> l2{1};
    assert(testOrder(l1, l2, Order::greater));
  }
  // Unordered
  if constexpr (std::is_same_v<Elem, PartialOrder>) {
    Container<Elem> l1{1, std::numeric_limits<int>::min()};
    Container<Elem> l2{1, 2};
    assert(testOrder(l1, l2, Order::unordered));
  }
}

// Tests the `operator<=>` on ordered containers
template <template <typename...> typename Container>
constexpr bool test_ordered_container_spaceship() {
  // The container should fulfil `std::three_way_comparable`
  static_assert(std::three_way_comparable<Container<int>>);

  // Test different comparison categories
  test_ordered_container_spaceship_with_type<Container, int, std::strong_ordering>();
  test_ordered_container_spaceship_with_type<Container, StrongOrder, std::strong_ordering>();
  test_ordered_container_spaceship_with_type<Container, WeakOrder, std::weak_ordering>();
  test_ordered_container_spaceship_with_type<Container, PartialOrder, std::partial_ordering>();

  // `LessAndEqComp` does not have `operator<=>`. ordering is sythesized based on `operator<`
  test_ordered_container_spaceship_with_type<Container, LessAndEqComp, std::weak_ordering>();

  // Thanks to SFINAE, the following is not a compiler error but returns `false`
  struct NonComparable {};
  static_assert(!std::three_way_comparable<Container<NonComparable>>);

  return true;
}

#endif
