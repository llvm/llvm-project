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

// iterator, const_iterator, local_iterator, const_local_iterator

#include <unordered_map>

#include <iterator>

using iterator = std::unordered_map<int, int>::iterator;
using const_iterator = std::unordered_map<int, int>::const_iterator;
using local_iterator = std::unordered_map<int, int>::local_iterator;
using const_local_iterator = std::unordered_map<int, int>::const_local_iterator;
using value_type = std::pair<const int, int>;

static_assert(std::forward_iterator<iterator>);
static_assert(!std::bidirectional_iterator<iterator>);
static_assert(!std::indirectly_writable<iterator, value_type>);
static_assert(std::sentinel_for<iterator, iterator>);
static_assert(std::sentinel_for<iterator, const_iterator>);
static_assert(!std::sentinel_for<iterator, local_iterator>);
static_assert(!std::sentinel_for<iterator, const_local_iterator>);
static_assert(!std::sized_sentinel_for<iterator, iterator>);
static_assert(!std::sized_sentinel_for<iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<iterator, local_iterator>);
static_assert(!std::sized_sentinel_for<iterator, const_local_iterator>);
static_assert(std::indirectly_movable<iterator, std::pair<int, int>*>);
static_assert(!std::indirectly_movable_storable<iterator, std::pair<int, int>*>);

static_assert(std::forward_iterator<const_iterator>);
static_assert(!std::bidirectional_iterator<const_iterator>);
static_assert(!std::indirectly_writable<const_iterator, value_type>);
static_assert(std::sentinel_for<const_iterator, iterator>);
static_assert(std::sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sentinel_for<const_iterator, local_iterator>);
static_assert(!std::sentinel_for<const_iterator, const_local_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, local_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_local_iterator>);

static_assert(std::forward_iterator<local_iterator>);
static_assert(!std::bidirectional_iterator<local_iterator>);
static_assert(!std::indirectly_writable<local_iterator, value_type>);
static_assert(!std::sentinel_for<local_iterator, iterator>);
static_assert(!std::sentinel_for<local_iterator, const_iterator>);
static_assert(std::sentinel_for<local_iterator, local_iterator>);
static_assert(std::sentinel_for<local_iterator, const_local_iterator>);
static_assert(!std::sized_sentinel_for<local_iterator, iterator>);
static_assert(!std::sized_sentinel_for<local_iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<local_iterator, local_iterator>);
static_assert(!std::sized_sentinel_for<local_iterator, const_local_iterator>);
static_assert(std::indirectly_movable<local_iterator, std::pair<int, int>*>);
static_assert(!std::indirectly_movable_storable<local_iterator, std::pair<int, int>*>);

static_assert(std::forward_iterator<const_local_iterator>);
static_assert(!std::bidirectional_iterator<const_local_iterator>);
static_assert(!std::indirectly_writable<const_local_iterator, value_type>);
static_assert(!std::sentinel_for<const_local_iterator, iterator>);
static_assert(!std::sentinel_for<const_local_iterator, const_iterator>);
static_assert(std::sentinel_for<const_local_iterator, local_iterator>);
static_assert(std::sentinel_for<const_local_iterator, const_local_iterator>);
static_assert(!std::sized_sentinel_for<const_local_iterator, iterator>);
static_assert(!std::sized_sentinel_for<const_local_iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<const_local_iterator, local_iterator>);
static_assert(!std::sized_sentinel_for<const_local_iterator, const_local_iterator>);
