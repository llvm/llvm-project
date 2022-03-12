//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <string_view>

#include <iterator>

#include "test_macros.h"

using iterator = std::string_view::iterator;
using const_iterator = std::string_view::const_iterator;
using reverse_iterator = std::string_view::reverse_iterator;
using const_reverse_iterator = std::string_view::const_reverse_iterator;

static_assert(std::contiguous_iterator<iterator>);
LIBCPP_STATIC_ASSERT(std::__is_cpp17_random_access_iterator<iterator>::value);
static_assert(!std::indirectly_writable<iterator, char>);
static_assert(std::sentinel_for<iterator, iterator>);
static_assert(std::sentinel_for<iterator, const_iterator>);
static_assert(!std::sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sentinel_for<iterator, const_reverse_iterator>);
static_assert(std::sized_sentinel_for<iterator, iterator>);
static_assert(std::sized_sentinel_for<iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<iterator, const_reverse_iterator>);
static_assert(std::indirectly_movable<iterator, char*>);
static_assert(std::indirectly_movable_storable<iterator, char*>);
static_assert(std::indirectly_copyable<iterator, char*>);
static_assert(std::indirectly_copyable_storable<iterator, char*>);
static_assert(!std::indirectly_swappable<iterator, iterator>);

static_assert(std::contiguous_iterator<const_iterator>);
LIBCPP_STATIC_ASSERT(std::__is_cpp17_random_access_iterator<const_iterator>::value);
static_assert(!std::indirectly_writable<const_iterator, char>);
static_assert(std::sentinel_for<const_iterator, iterator>);
static_assert(std::sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(std::sized_sentinel_for<const_iterator, iterator>);
static_assert(std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(!std::indirectly_swappable<const_iterator, const_iterator>);
