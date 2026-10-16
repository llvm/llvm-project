//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <flat_set>
#include <deque>
#include <functional>
#include <iterator>
#include <string>
#include <vector>
#include <type_traits>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using C   = std::flat_set<Key, std::less<Key>, KeyContainer>;
  using I   = C::iterator;
  using CI  = C::const_iterator;
  using RI  = C::reverse_iterator;
  using CRI = C::const_reverse_iterator;
  static_assert(std::random_access_iterator<I>);
  static_assert(std::random_access_iterator<CI>);
  static_assert(std::random_access_iterator<RI>);
  static_assert(std::random_access_iterator<CRI>);
  static_assert(!std::contiguous_iterator<RI>);
  static_assert(!std::contiguous_iterator<CRI>);
  static_assert(!std::indirectly_writable<I, std::pair<int, char>>);
  static_assert(!std::indirectly_writable<CI, std::pair<int, char>>);
  static_assert(!std::indirectly_writable<RI, std::pair<int, char>>);
  static_assert(!std::indirectly_writable<CRI, std::pair<int, char>>);
  static_assert(std::sentinel_for<I, I>);
  static_assert(std::sentinel_for<I, CI>);
  static_assert(!std::sentinel_for<I, RI>);
  static_assert(!std::sentinel_for<I, CRI>);
  static_assert(std::sentinel_for<CI, I>);
  static_assert(std::sentinel_for<CI, CI>);
  static_assert(!std::sentinel_for<CI, RI>);
  static_assert(!std::sentinel_for<CI, CRI>);
  static_assert(!std::sentinel_for<RI, I>);
  static_assert(!std::sentinel_for<RI, CI>);
  static_assert(std::sentinel_for<RI, RI>);
  static_assert(std::sentinel_for<RI, CRI>);
  static_assert(!std::sentinel_for<CRI, I>);
  static_assert(!std::sentinel_for<CRI, CI>);
  static_assert(std::sentinel_for<CRI, RI>);
  static_assert(std::sentinel_for<CRI, CRI>);
  static_assert(std::indirectly_movable_storable<I, Key*>);
  static_assert(std::indirectly_movable_storable<CI, Key*>);
  static_assert(std::indirectly_movable_storable<RI, Key*>);
  static_assert(std::indirectly_movable_storable<CRI, Key*>);

  static_assert(std::is_same_v<typename std::iterator_traits<I>::iterator_category, std::random_access_iterator_tag>);
  static_assert(std::is_same_v<typename std::iterator_traits<CI>::iterator_category, std::random_access_iterator_tag>);
  static_assert(std::is_same_v<typename std::iterator_traits<RI>::iterator_category, std::random_access_iterator_tag>);
  static_assert(std::is_same_v<typename std::iterator_traits<CRI>::iterator_category, std::random_access_iterator_tag>);
}

void test() {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();
}
