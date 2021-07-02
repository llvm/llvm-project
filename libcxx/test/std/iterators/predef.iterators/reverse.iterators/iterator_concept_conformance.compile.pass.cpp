//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// reverse_iterator

#include <iterator>

#include "test_iterators.h"

template<class I1>
constexpr bool common_reverse_iterator_checks() {
  static_assert(std::indirectly_writable<I1, int>);
  static_assert(std::sentinel_for<I1, I1>);
  static_assert(std::sentinel_for<I1, std::reverse_iterator<float*>>);
  static_assert(!std::sized_sentinel_for<I1, std::reverse_iterator<float*>>);
  return true;
}

using reverse_bidirectional_iterator = std::reverse_iterator<bidirectional_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_bidirectional_iterator>());
static_assert(std::bidirectional_iterator<reverse_bidirectional_iterator>);
static_assert(!std::random_access_iterator<reverse_bidirectional_iterator>);
static_assert(!std::sized_sentinel_for<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert( std::indirectly_movable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert( std::indirectly_movable_storable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);
static_assert( std::indirectly_swappable<reverse_bidirectional_iterator, reverse_bidirectional_iterator>);

using reverse_random_access_iterator = std::reverse_iterator<random_access_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_random_access_iterator>());
static_assert(std::random_access_iterator<reverse_random_access_iterator>);
static_assert(!std::contiguous_iterator<reverse_random_access_iterator>);
static_assert(std::sized_sentinel_for<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert( std::indirectly_movable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert( std::indirectly_movable_storable<reverse_random_access_iterator, reverse_random_access_iterator>);
static_assert( std::indirectly_swappable<reverse_random_access_iterator, reverse_random_access_iterator>);

using reverse_contiguous_iterator = std::reverse_iterator<contiguous_iterator<int*>>;
static_assert(common_reverse_iterator_checks<reverse_contiguous_iterator>());
static_assert(std::random_access_iterator<reverse_contiguous_iterator>);
static_assert(!std::contiguous_iterator<reverse_contiguous_iterator>);
static_assert(std::sized_sentinel_for<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert( std::indirectly_movable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert( std::indirectly_movable_storable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
static_assert( std::indirectly_swappable<reverse_contiguous_iterator, reverse_contiguous_iterator>);
