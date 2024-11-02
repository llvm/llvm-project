//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// unordered_multimap

#include <unordered_map>

#include <concepts>
#include <ranges>

#include "test_macros.h"

using range = std::unordered_multimap<int, int>;


static_assert(std::same_as<std::ranges::iterator_t<range>, range::iterator>);
static_assert(std::ranges::common_range<range>);
static_assert(std::ranges::forward_range<range>);
LIBCPP_STATIC_ASSERT(!std::ranges::bidirectional_range<range>);
static_assert(!std::ranges::view<range>);
static_assert(std::ranges::sized_range<range>);
static_assert(!std::ranges::borrowed_range<range>);
static_assert(std::ranges::viewable_range<range>);

static_assert(std::same_as<std::ranges::iterator_t<range const>, range::const_iterator>);
static_assert(std::ranges::common_range<range const>);
static_assert(std::ranges::forward_range<range const>);
LIBCPP_STATIC_ASSERT(!std::ranges::bidirectional_range<range const>);
static_assert(!std::ranges::view<range const>);
static_assert(std::ranges::sized_range<range const>);
static_assert(!std::ranges::borrowed_range<range const>);
static_assert(!std::ranges::viewable_range<range const>);
