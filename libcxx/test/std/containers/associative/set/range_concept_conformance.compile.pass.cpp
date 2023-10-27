//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// set

#include <set>

#include <concepts>
#include <ranges>

using range = std::set<int>;


static_assert(std::same_as<std::ranges::iterator_t<range>, range::iterator>);
static_assert(std::ranges::bidirectional_range<range>);
static_assert(!std::ranges::random_access_range<range>);
static_assert(std::ranges::common_range<range>);
static_assert(std::ranges::input_range<range>);
static_assert(!std::ranges::view<range>);
static_assert(std::ranges::sized_range<range>);
static_assert(!std::ranges::borrowed_range<range>);
static_assert(std::ranges::viewable_range<range>);

static_assert(std::same_as<std::ranges::iterator_t<range const>, range::const_iterator>);
static_assert(std::ranges::bidirectional_range<range const>);
static_assert(!std::ranges::random_access_range<range const>);
static_assert(std::ranges::common_range<range const>);
static_assert(std::ranges::input_range<range const>);
static_assert(!std::ranges::view<range const>);
static_assert(std::ranges::sized_range<range const>);
static_assert(!std::ranges::borrowed_range<range const>);
static_assert(!std::ranges::viewable_range<range const>);
