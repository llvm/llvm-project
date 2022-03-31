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

using iterator = std::span<int>::iterator;
using reverse_iterator = std::span<int>::reverse_iterator;
using value_type = int;

static_assert(std::contiguous_iterator<iterator>);
LIBCPP_STATIC_ASSERT(std::__is_cpp17_random_access_iterator<iterator>::value);
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
