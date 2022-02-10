//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I>
//   concept permutable = see below; // Since C++20

#include <iterator>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

using AllConstraintsSatisfied = forward_iterator<int*>;
static_assert( std::forward_iterator<AllConstraintsSatisfied>);
static_assert( std::indirectly_movable_storable<AllConstraintsSatisfied, AllConstraintsSatisfied>);
static_assert( std::indirectly_swappable<AllConstraintsSatisfied>);
static_assert( std::permutable<AllConstraintsSatisfied>);

using NotAForwardIterator = cpp20_input_iterator<int*>;
static_assert(!std::forward_iterator<NotAForwardIterator>);
static_assert( std::indirectly_movable_storable<NotAForwardIterator, NotAForwardIterator>);
static_assert( std::indirectly_swappable<NotAForwardIterator>);
static_assert(!std::permutable<NotAForwardIterator>);

struct NonCopyable {
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  friend void swap(NonCopyable&, NonCopyable&);
};
using NotIMS = forward_iterator<NonCopyable*>;

static_assert( std::forward_iterator<NotIMS>);
static_assert(!std::indirectly_movable_storable<NotIMS, NotIMS>);
static_assert( std::indirectly_swappable<NotIMS>);
static_assert(!std::permutable<NotIMS>);

// Note: it is impossible for an iterator to satisfy `indirectly_movable_storable` but not `indirectly_swappable`:
// `indirectly_swappable` requires both iterators to be `indirectly_readable` and for `ranges::iter_swap` to be
// well-formed for both iterators. `indirectly_movable_storable` also requires the iterator to be `indirectly_readable`.
// `ranges::iter_swap` is always defined for `indirectly_movable_storable` iterators.
