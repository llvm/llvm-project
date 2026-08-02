//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models forward_range:
//     class iterator;

//     using iterator::iterator_category = input_iterator_tag;
//     using iterator::iterator_concept = see below;
//     using iterator::value_type = decltype(views::take(subrange(current_, end_), n_));
//     using iterator::difference_type = range_difference_t<Base>;

#include <concepts>
#include <iterator>
#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

template <template <class...> class Iter>
using ChunkViewFor = std::ranges::chunk_view<test_view<Iter>>;

template <template <class...> class Iter>
using ChunkIteratorFor = std::ranges::iterator_t<ChunkViewFor<Iter>>;

template <template <class...> class Iter>
constexpr void test_iterator_types() {
  using ChunkView     = ChunkViewFor<Iter>;
  using ChunkIterator = ChunkIteratorFor<Iter>;

  static_assert(std::same_as<typename ChunkIterator::iterator_category, std::input_iterator_tag>);
  static_assert(std::same_as<typename ChunkIterator::value_type, std::ranges::range_value_t<ChunkView>>);
  static_assert(
      std::same_as<typename ChunkIterator::difference_type, std::ranges::range_difference_t<test_view<Iter>>>);
}

constexpr void test() {
  test_iterator_types<forward_iterator>();
  test_iterator_types<bidirectional_iterator>();
  test_iterator_types<random_access_iterator>();
  test_iterator_types<contiguous_iterator>();

  static_assert(std::same_as<ChunkIteratorFor<forward_iterator>::iterator_concept, std::forward_iterator_tag>);
  static_assert(
      std::same_as<ChunkIteratorFor<bidirectional_iterator>::iterator_concept, std::bidirectional_iterator_tag>);
  static_assert(
      std::same_as<ChunkIteratorFor<random_access_iterator>::iterator_concept, std::random_access_iterator_tag>);
  static_assert(std::same_as<ChunkIteratorFor<contiguous_iterator>::iterator_concept, std::random_access_iterator_tag>);
}
