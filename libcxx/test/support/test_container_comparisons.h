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

// Implementation detail of `test_sequence_container_spaceship`
template <template <typename...> typename Container, typename Elem, typename Order>
constexpr void test_sequence_container_spaceship_with_type() {
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

// Tests the `operator<=>` on sequence containers
template <template <typename...> typename Container>
constexpr bool test_sequence_container_spaceship() {
  // The container should fulfill `std::three_way_comparable`
  static_assert(std::three_way_comparable<Container<int>>);

  // Test different comparison categories
  test_sequence_container_spaceship_with_type<Container, int, std::strong_ordering>();
  test_sequence_container_spaceship_with_type<Container, StrongOrder, std::strong_ordering>();
  test_sequence_container_spaceship_with_type<Container, WeakOrder, std::weak_ordering>();
  test_sequence_container_spaceship_with_type<Container, PartialOrder, std::partial_ordering>();

  // `LessAndEqComp` does not have `operator<=>`. Ordering is synthesized based on `operator<`
  test_sequence_container_spaceship_with_type<Container, LessAndEqComp, std::weak_ordering>();

  // Thanks to SFINAE, the following is not a compiler error but returns `false`
  struct NonComparable {};
  static_assert(!std::three_way_comparable<Container<NonComparable>>);

  return true;
}

// Implementation detail of `test_ordered_map_container_spaceship`
template <template <typename...> typename Container, typename Key, typename Val, typename Order>
constexpr void test_ordered_map_container_spaceship_with_type() {
  // Empty containers
  {
    Container<Key, Val> l1;
    Container<Key, Val> l2;
    assert(testOrder(l1, l2, Order::equivalent));
  }
  // Identical contents
  {
    Container<Key, Val> l1{{1, 1}, {2, 1}};
    Container<Key, Val> l2{{1, 1}, {2, 1}};
    assert(testOrder(l1, l2, Order::equivalent));
  }
  // Less, due to contained values
  {
    Container<Key, Val> l1{{1, 1}, {2, 1}};
    Container<Key, Val> l2{{1, 1}, {2, 2}};
    assert(testOrder(l1, l2, Order::less));
  }
  // Greater, due to contained values
  {
    Container<Key, Val> l1{{1, 1}, {2, 3}};
    Container<Key, Val> l2{{1, 1}, {2, 2}};
    assert(testOrder(l1, l2, Order::greater));
  }
  // Shorter list
  {
    Container<Key, Val> l1{{1, 1}};
    Container<Key, Val> l2{{1, 1}, {2, 2}};
    assert(testOrder(l1, l2, Order::less));
  }
  // Longer list
  {
    Container<Key, Val> l1{{1, 2}, {2, 2}};
    Container<Key, Val> l2{{1, 1}};
    assert(testOrder(l1, l2, Order::greater));
  }
  // Unordered
  if constexpr (std::is_same_v<Val, PartialOrder>) {
    Container<Key, Val> l1{{1, 1}, {2, std::numeric_limits<int>::min()}};
    Container<Key, Val> l2{{1, 1}, {2, 2}};
    assert(testOrder(l1, l2, Order::unordered));
  }

  // Identical contents
  {
    Container<Key, Val> l1{{1, 1}, {2, 1}, {2, 2}};
    Container<Key, Val> l2{{1, 1}, {2, 1}, {2, 2}};
    assert(testOrder(l1, l2, Order::equivalent));
    Container<Key, Val> l3{{1, 1}, {2, 1}, {2, 2}};
    Container<Key, Val> l4{{2, 1}, {2, 2}, {1, 1}};
    assert(testOrder(l3, l4, Order::equivalent));
  }
  // Less, due to contained values
  {
    Container<Key, Val> l1{{1, 1}, {2, 1}, {2, 1}};
    Container<Key, Val> l2{{1, 1}, {2, 2}, {2, 2}};
    assert(testOrder(l1, l2, Order::less));
    Container<Key, Val> l3{{1, 1}, {2, 1}, {2, 1}};
    Container<Key, Val> l4{{2, 2}, {2, 2}, {1, 1}};
    assert(testOrder(l3, l4, Order::less));
  }
  // Greater, due to contained values
  {
    Container<Key, Val> l1{{1, 1}, {2, 3}, {2, 3}};
    Container<Key, Val> l2{{1, 1}, {2, 2}, {2, 2}};
    assert(testOrder(l1, l2, Order::greater));
    Container<Key, Val> l3{{1, 1}, {2, 3}, {2, 3}};
    Container<Key, Val> l4{{2, 2}, {2, 2}, {1, 1}};
    assert(testOrder(l3, l4, Order::greater));
  }
  // Shorter list
  {
    Container<Key, Val> l1{{1, 1}, {2, 2}};
    Container<Key, Val> l2{{1, 1}, {2, 2}, {2, 2}, {3, 1}};
    assert(testOrder(l1, l2, Order::less));
    Container<Key, Val> l3{{1, 1}, {2, 2}};
    Container<Key, Val> l4{{3, 1}, {2, 2}, {2, 2}, {1, 1}};
    assert(testOrder(l3, l4, Order::less));
  }
  // Longer list
  {
    Container<Key, Val> l1{{1, 2}, {2, 2}, {2, 2}, {3, 1}};
    Container<Key, Val> l2{{1, 1}, {2, 2}};
    assert(testOrder(l1, l2, Order::greater));
    Container<Key, Val> l3{{1, 2}, {2, 2}, {2, 2}, {3, 1}};
    Container<Key, Val> l4{{2, 2}, {1, 1}};
    assert(testOrder(l3, l4, Order::greater));
  }
  // Unordered
  if constexpr (std::is_same_v<Val, PartialOrder>) {
    Container<Key, Val> l1{{1, 1}, {2, std::numeric_limits<int>::min()}, {2, 3}};
    Container<Key, Val> l2{{1, 1}, {2, 2}, {2, 3}};
    assert(testOrder(l1, l2, Order::unordered));
    Container<Key, Val> l3{{1, 1}, {2, std::numeric_limits<int>::min()}, {2, 3}};
    Container<Key, Val> l4{{2, 3}, {2, 2}, {1, 1}};
    assert(testOrder(l3, l4, Order::unordered));
  }
}

// Tests the `operator<=>` on ordered containers
template <template <typename...> typename Container>
constexpr bool test_ordered_map_container_spaceship() {
  // The container should fulfill `std::three_way_comparable`
  static_assert(std::three_way_comparable<Container<int, int>>);

  // Test different comparison categories
  test_ordered_map_container_spaceship_with_type<Container, int, int, std::strong_ordering>();
  test_ordered_map_container_spaceship_with_type<Container, int, StrongOrder, std::strong_ordering>();
  test_ordered_map_container_spaceship_with_type<Container, int, WeakOrder, std::weak_ordering>();
  test_ordered_map_container_spaceship_with_type<Container, int, PartialOrder, std::partial_ordering>();

  // `LessAndEqComp` does not have `operator<=>`. Ordering is synthesized based on `operator<`
  test_ordered_map_container_spaceship_with_type<Container, int, LessAndEqComp, std::weak_ordering>();

  // Thanks to SFINAE, the following is not a compiler error but returns `false`
  struct NonComparable {};
  static_assert(!std::three_way_comparable<Container<int, NonComparable>>);

  return true;
}

#endif
