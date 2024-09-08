//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <string>

#include <iterator>

#include "nasty_string.h"

template <class String>
void test() {
  using iterator               = typename String::iterator;
  using const_iterator         = typename String::const_iterator;
  using reverse_iterator       = typename String::reverse_iterator;
  using const_reverse_iterator = typename String::const_reverse_iterator;
  using value_type             = typename String::value_type;

  static_assert(std::contiguous_iterator<iterator>);
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

void tests() {
  test<std::string>();
  test<std::wstring>();
#if !defined(TEST_HAS_NO_NASTY_STRING)
  test<nasty_string>();
#endif
}
