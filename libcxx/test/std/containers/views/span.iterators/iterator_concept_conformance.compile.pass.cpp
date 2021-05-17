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

// iterator, reverse_iterator

#include <span>

#include <iterator>

using iterator = std::span<int>::iterator;
using reverse_iterator = std::span<int>::reverse_iterator;
using value_type = int;

static_assert(std::contiguous_iterator<iterator>);
static_assert(std::indirectly_writable<iterator, value_type>);
static_assert(std::sentinel_for<iterator, iterator>);
static_assert(!std::sentinel_for<iterator, reverse_iterator>);
static_assert(std::sized_sentinel_for<iterator, iterator>);
static_assert(!std::sized_sentinel_for<iterator, reverse_iterator>);
