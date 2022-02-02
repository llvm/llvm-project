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

#include <iterator>
#include <vector>

#include "test_macros.h"

using iterator = std::vector<bool>::iterator;
using const_iterator = std::vector<bool>::const_iterator;
using reverse_iterator = std::vector<bool>::reverse_iterator;
using const_reverse_iterator = std::vector<bool>::const_reverse_iterator;
using value_type = bool;

static_assert( std::random_access_iterator<iterator>);
static_assert( std::random_access_iterator<reverse_iterator>);
static_assert(!std::contiguous_iterator<iterator>);
static_assert(!std::contiguous_iterator<reverse_iterator>);
#if TEST_STD_VER > 20
static_assert( std::indirectly_writable<iterator, value_type>);
#else
static_assert(!std::indirectly_writable<iterator, value_type>);
#endif
static_assert( std::sentinel_for<iterator, iterator>);
static_assert( std::sentinel_for<iterator, const_iterator>);
static_assert(!std::sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sentinel_for<iterator, const_reverse_iterator>);
static_assert( std::sized_sentinel_for<iterator, iterator>);
static_assert( std::sized_sentinel_for<iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<iterator, const_reverse_iterator>);
static_assert(std::indirectly_movable<iterator, bool*>);
static_assert(std::indirectly_movable_storable<iterator, bool*>);
static_assert(std::indirectly_copyable<iterator, bool*>);
static_assert(std::indirectly_copyable_storable<iterator, bool*>);
static_assert(std::indirectly_swappable<iterator, iterator>);

static_assert( std::random_access_iterator<const_iterator>);
static_assert( std::random_access_iterator<const_reverse_iterator>);
static_assert(!std::contiguous_iterator<const_iterator>);
static_assert(!std::contiguous_iterator<const_reverse_iterator>);
static_assert(!std::indirectly_writable<const_iterator, value_type>);
static_assert( std::sentinel_for<const_iterator, iterator>);
static_assert( std::sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sentinel_for<const_iterator, const_reverse_iterator>);
static_assert( std::sized_sentinel_for<const_iterator, iterator>);
static_assert( std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(!std::indirectly_swappable<const_iterator, const_iterator>);
