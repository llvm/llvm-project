//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// iterator, reverse_iterator

#include <span>

#include <iterator>
#include "test_macros.h"

using iterator         = std::span<int>::iterator;
using reverse_iterator = std::span<int>::reverse_iterator;
using value_type       = int;

#if TEST_STD_VER >= 23
using const_iterator         = std::span<int>::const_iterator;
using const_reverse_iterator = std::span<int>::const_reverse_iterator;
#endif

static_assert(std::contiguous_iterator<iterator>);
LIBCPP_STATIC_ASSERT(std::__has_random_access_iterator_category<iterator>::value);
static_assert(std::indirectly_writable<iterator, value_type>);
static_assert(std::sentinel_for<iterator, iterator>);
static_assert(!std::sentinel_for<iterator, reverse_iterator>);
static_assert(std::sized_sentinel_for<iterator, iterator>);
static_assert(!std::sized_sentinel_for<iterator, reverse_iterator>);
static_assert(std::indirectly_movable<iterator, int*>);
static_assert(std::indirectly_movable_storable<iterator, int*>);
static_assert(std::indirectly_copyable<iterator, int*>);
static_assert(std::indirectly_copyable_storable<iterator, int*>);
static_assert(std::indirectly_swappable<iterator, iterator>);

#if TEST_STD_VER >= 23
static_assert(std::contiguous_iterator<const_iterator>);
LIBCPP_STATIC_ASSERT(std::__has_random_access_iterator_category<const_iterator>::value);
static_assert(std::sentinel_for<const_iterator, const_iterator>);
static_assert(std::sentinel_for<const_iterator, iterator>);
static_assert(std::sentinel_for<iterator, const_iterator>);
static_assert(!std::sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(!std::sentinel_for<const_iterator, reverse_iterator>);
static_assert(std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(std::sized_sentinel_for<iterator, const_iterator>);
static_assert(std::sized_sentinel_for<const_iterator, iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, reverse_iterator>);
#endif
