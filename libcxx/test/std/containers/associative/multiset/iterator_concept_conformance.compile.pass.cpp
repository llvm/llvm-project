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
// XFAIL: msvc && clang

// iterator, const_iterator, reverse_iterator, const_reverse_iterator

#include <set>

#include <iterator>

using iterator = std::multiset<int>::iterator;
using const_iterator = std::multiset<int>::const_iterator;
using reverse_iterator = std::multiset<int>::reverse_iterator;
using const_reverse_iterator = std::multiset<int>::const_reverse_iterator;
using value_type = int;

static_assert(std::indirectly_readable<iterator>);
static_assert(!std::indirectly_writable<iterator, value_type>);
static_assert(std::incrementable<iterator>);
static_assert(std::input_or_output_iterator<iterator>);
static_assert(std::sentinel_for<iterator, iterator>);
static_assert(std::sentinel_for<iterator, const_iterator>);
static_assert(!std::sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sentinel_for<iterator, const_reverse_iterator>);
static_assert(!std::sized_sentinel_for<iterator, iterator>);
static_assert(!std::sized_sentinel_for<iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<iterator, const_reverse_iterator>);

static_assert(std::indirectly_readable<const_iterator>);
static_assert(!std::indirectly_writable<const_iterator, value_type>);
static_assert(std::incrementable<const_iterator>);
static_assert(std::sentinel_for<const_iterator, iterator>);
static_assert(!std::sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sentinel_for<const_iterator, const_reverse_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, reverse_iterator>);
static_assert(!std::sized_sentinel_for<const_iterator, const_reverse_iterator>);
