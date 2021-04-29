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

// iterator, const_iterator, local_iterator, const_local_iterator

#include <unordered_set>

#include <iterator>

using iterator = std::unordered_set<int>::iterator;
using const_iterator = std::unordered_set<int>::const_iterator;
using local_iterator = std::unordered_set<int>::local_iterator;
using const_local_iterator = std::unordered_set<int>::const_local_iterator;
using value_type = int;

static_assert(std::indirectly_readable<iterator>);
static_assert(!std::indirectly_writable<iterator, value_type>);
static_assert(std::incrementable<iterator>);
static_assert(std::input_or_output_iterator<iterator>);
static_assert(std::sentinel_for<iterator, iterator>);
static_assert(std::sentinel_for<iterator, const_iterator>);
static_assert(!std::sentinel_for<iterator, local_iterator>);
static_assert(!std::sentinel_for<iterator, const_local_iterator>);

static_assert(std::indirectly_readable<const_iterator>);
static_assert(!std::indirectly_writable<const_iterator, value_type>);
static_assert(std::incrementable<const_iterator>);
static_assert(std::sentinel_for<const_iterator, iterator>);
static_assert(!std::sentinel_for<const_iterator, local_iterator>);
static_assert(!std::sentinel_for<const_iterator, const_local_iterator>);

static_assert(std::indirectly_readable<local_iterator>);
static_assert(std::incrementable<local_iterator>);
static_assert(std::input_or_output_iterator<local_iterator>);
static_assert(!std::sentinel_for<local_iterator, iterator>);
static_assert(!std::sentinel_for<local_iterator, const_iterator>);
static_assert(std::sentinel_for<local_iterator, local_iterator>);
static_assert(std::sentinel_for<local_iterator, const_local_iterator>);

static_assert(std::indirectly_readable<const_local_iterator>);
static_assert(!std::indirectly_writable<const_local_iterator, value_type>);
static_assert(std::incrementable<const_local_iterator>);
static_assert(std::input_or_output_iterator<const_local_iterator>);
static_assert(!std::sentinel_for<const_local_iterator, iterator>);
static_assert(!std::sentinel_for<const_local_iterator, const_iterator>);
static_assert(std::sentinel_for<const_local_iterator, local_iterator>);
static_assert(std::sentinel_for<const_local_iterator, const_local_iterator>);
