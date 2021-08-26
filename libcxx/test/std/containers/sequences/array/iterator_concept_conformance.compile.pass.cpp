//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <array>

#include <iterator>

using iterator = std::array<int, 10>::iterator;
using const_iterator = std::array<int, 10>::const_iterator;
using reverse_iterator = std::array<int, 10>::reverse_iterator;
using const_reverse_iterator = std::array<int, 10>::const_reverse_iterator;

static_assert(std::contiguous_iterator<iterator>);
static_assert(std::indirectly_writable<iterator, int>);
static_assert(std::sentinel_for<iterator, iterator>);
static_assert(std::sentinel_for<iterator, const_iterator>);
static_assert(!std::sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sentinel_for<iterator, const_reverse_iterator>);
static_assert(std::sized_sentinel_for<iterator, iterator>);
static_assert(std::sized_sentinel_for<iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<iterator, const_reverse_iterator>);
static_assert( std::indirectly_movable<iterator, iterator>);
static_assert( std::indirectly_movable_storable<iterator, iterator>);
static_assert(!std::indirectly_movable<iterator, const_iterator>);
static_assert(!std::indirectly_movable_storable<iterator, const_iterator>);
static_assert( std::indirectly_movable<iterator, reverse_iterator>);
static_assert( std::indirectly_movable_storable<iterator, reverse_iterator>);
static_assert(!std::indirectly_movable<iterator, const_reverse_iterator>);
static_assert(!std::indirectly_movable_storable<iterator, const_reverse_iterator>);
static_assert( std::indirectly_swappable<iterator, iterator>);

static_assert(std::contiguous_iterator<const_iterator>);
static_assert(!std::indirectly_writable<const_iterator, int>);
static_assert(std::sentinel_for<const_iterator, iterator>);
static_assert(std::sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(std::sized_sentinel_for<const_iterator, iterator>);
static_assert(std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
static_assert( std::indirectly_movable<const_iterator, iterator>);
static_assert( std::indirectly_movable_storable<const_iterator, iterator>);
static_assert(!std::indirectly_movable<const_iterator, const_iterator>);
static_assert(!std::indirectly_movable_storable<const_iterator, const_iterator>);
static_assert( std::indirectly_movable<const_iterator, reverse_iterator>);
static_assert( std::indirectly_movable_storable<const_iterator, reverse_iterator>);
static_assert(!std::indirectly_movable<const_iterator, const_reverse_iterator>);
static_assert(!std::indirectly_movable_storable<const_iterator, const_reverse_iterator>);
static_assert(!std::indirectly_swappable<const_iterator, const_iterator>);
