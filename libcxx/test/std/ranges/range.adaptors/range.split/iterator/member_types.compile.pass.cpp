//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Member typedefs in split_view<V, P>::iterator.

#include <concepts>
#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter, class PatternIter>
constexpr void testIteratorTypedef() {
  using Range     = std::ranges::subrange<Iter, sentinel_wrapper<Iter>>;
  using Pattern   = std::ranges::subrange<PatternIter, sentinel_wrapper<PatternIter>>;
  using SplitIter = std::ranges::iterator_t<std::ranges::split_view<Range, Pattern>>;

  static_assert(std::same_as<typename SplitIter::iterator_concept, //
                             std::forward_iterator_tag>);

  static_assert(std::same_as<typename SplitIter::iterator_category, //
                             std::input_iterator_tag>);

  static_assert(std::same_as<typename SplitIter::value_type, //
                             std::ranges::subrange<Iter>>);

  static_assert(std::same_as<typename SplitIter::difference_type, //
                             std::iter_difference_t<Iter>>);
}

template <class Iter>
void testIteratorTypedefPattern() {
  testIteratorTypedef<Iter, forward_iterator<int*>>();
  testIteratorTypedef<Iter, bidirectional_iterator<int*>>();
  testIteratorTypedef<Iter, random_access_iterator<int*>>();
  testIteratorTypedef<Iter, contiguous_iterator<int*>>();
  testIteratorTypedef<Iter, int*>();
}

void test() {
  testIteratorTypedefPattern<forward_iterator<int*>>();
  testIteratorTypedefPattern<bidirectional_iterator<int*>>();
  testIteratorTypedefPattern<random_access_iterator<int*>>();
  testIteratorTypedefPattern<contiguous_iterator<int*>>();
  testIteratorTypedefPattern<int*>();
}
