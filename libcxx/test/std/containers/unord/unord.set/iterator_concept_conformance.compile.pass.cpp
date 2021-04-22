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

#include <unordered_set>

#include <iterator>

using iterator = std::unordered_set<int>::iterator;
using const_iterator = std::unordered_set<int>::const_iterator;
using local_iterator = std::unordered_set<int>::local_iterator;
using const_local_iterator = std::unordered_set<int>::const_local_iterator;
using value_type = iterator::value_type;

static_assert(std::indirectly_readable<iterator>);
static_assert(!std::indirectly_writable<iterator, value_type>);

static_assert(std::indirectly_readable<const_iterator>);
static_assert(!std::indirectly_writable<const_iterator, value_type>);

static_assert(std::indirectly_readable<local_iterator>);
static_assert(!std::indirectly_writable<local_iterator, value_type>);

static_assert(std::indirectly_readable<const_local_iterator>);
static_assert(!std::indirectly_writable<const_local_iterator, value_type>);
