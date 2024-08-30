//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <inplace_vector>

#include <iterator>

template <typename T, std::size_t N>
void test() {
  using V                      = std::inplace_vector<T, N>;
  using iterator               = typename V::iterator;
  using const_iterator         = typename V::const_iterator;
  using reverse_iterator       = typename V::reverse_iterator;
  using const_reverse_iterator = typename V::const_reverse_iterator;
  using value_type             = T;

  static_assert(std::contiguous_iterator<iterator>);
  static_assert(std::random_access_iterator<reverse_iterator>);
  static_assert(!std::contiguous_iterator<reverse_iterator>);
  static_assert(std::indirectly_writable<iterator, value_type>);
  static_assert(std::sentinel_for<iterator, iterator>);
  static_assert(std::sentinel_for<iterator, const_iterator>);
  static_assert(!std::sentinel_for<iterator, reverse_iterator>);
  static_assert(!std::sentinel_for<iterator, const_reverse_iterator>);
  static_assert(std::sized_sentinel_for<iterator, iterator>);
  static_assert(std::sized_sentinel_for<iterator, const_iterator>);
  static_assert(!std::sized_sentinel_for<iterator, reverse_iterator>);
  static_assert(!std::sized_sentinel_for<iterator, const_reverse_iterator>);
  static_assert(std::indirectly_movable<iterator, iterator>);
  static_assert(std::indirectly_movable_storable<iterator, iterator>);
  static_assert(!std::indirectly_movable<iterator, const_iterator>);
  static_assert(!std::indirectly_movable_storable<iterator, const_iterator>);
  static_assert(std::indirectly_movable<iterator, reverse_iterator>);
  static_assert(std::indirectly_movable_storable<iterator, reverse_iterator>);
  static_assert(!std::indirectly_movable<iterator, const_reverse_iterator>);
  static_assert(!std::indirectly_movable_storable<iterator, const_reverse_iterator>);
  static_assert(std::indirectly_copyable<iterator, iterator>);
  static_assert(std::indirectly_copyable_storable<iterator, iterator>);
  static_assert(!std::indirectly_copyable<iterator, const_iterator>);
  static_assert(!std::indirectly_copyable_storable<iterator, const_iterator>);
  static_assert(std::indirectly_copyable<iterator, reverse_iterator>);
  static_assert(std::indirectly_copyable_storable<iterator, reverse_iterator>);
  static_assert(!std::indirectly_copyable<iterator, const_reverse_iterator>);
  static_assert(!std::indirectly_copyable_storable<iterator, const_reverse_iterator>);
  static_assert(std::indirectly_swappable<iterator, iterator>);

  static_assert(std::contiguous_iterator<const_iterator>);
  static_assert(std::random_access_iterator<const_reverse_iterator>);
  static_assert(!std::contiguous_iterator<const_reverse_iterator>);
  static_assert(!std::indirectly_writable<const_iterator, value_type>);
  static_assert(std::sentinel_for<const_iterator, iterator>);
  static_assert(std::sentinel_for<const_iterator, const_iterator>);
  static_assert(!std::sentinel_for<const_iterator, reverse_iterator>);
  static_assert(!std::sentinel_for<const_iterator, const_reverse_iterator>);
  static_assert(std::sized_sentinel_for<const_iterator, iterator>);
  static_assert(std::sized_sentinel_for<const_iterator, const_iterator>);
  static_assert(!std::sized_sentinel_for<const_iterator, reverse_iterator>);
  static_assert(!std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
  static_assert(std::indirectly_movable<const_iterator, iterator>);
  static_assert(std::indirectly_movable_storable<const_iterator, iterator>);
  static_assert(!std::indirectly_movable<const_iterator, const_iterator>);
  static_assert(!std::indirectly_movable_storable<const_iterator, const_iterator>);
  static_assert(std::indirectly_movable<const_iterator, reverse_iterator>);
  static_assert(std::indirectly_movable_storable<const_iterator, reverse_iterator>);
  static_assert(!std::indirectly_movable<const_iterator, const_reverse_iterator>);
  static_assert(!std::indirectly_movable_storable<const_iterator, const_reverse_iterator>);
  static_assert(std::indirectly_copyable<const_iterator, iterator>);
  static_assert(std::indirectly_copyable_storable<const_iterator, iterator>);
  static_assert(!std::indirectly_copyable<const_iterator, const_iterator>);
  static_assert(!std::indirectly_copyable_storable<const_iterator, const_iterator>);
  static_assert(std::indirectly_copyable<const_iterator, reverse_iterator>);
  static_assert(std::indirectly_copyable_storable<const_iterator, reverse_iterator>);
  static_assert(!std::indirectly_copyable<const_iterator, const_reverse_iterator>);
  static_assert(!std::indirectly_copyable_storable<const_iterator, const_reverse_iterator>);
  static_assert(!std::indirectly_swappable<const_iterator, const_iterator>);
}

struct NonTrivial {
  int i = 0;
};

void tests() {
  test<int, 0>();
  test<int, 10>();
  test<NonTrivial, 0>();
  test<NonTrivial, 10>();
}
