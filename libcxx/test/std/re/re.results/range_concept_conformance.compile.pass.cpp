//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// match_results

#include <regex>

#include <concepts>
#include <ranges>



static_assert(std::same_as<std::ranges::iterator_t<std::cmatch>, std::cmatch::iterator>);
static_assert(std::ranges::common_range<std::cmatch>);
static_assert(std::ranges::random_access_range<std::cmatch>);
static_assert(std::ranges::contiguous_range<std::cmatch>);
static_assert(!std::ranges::view<std::cmatch>);
static_assert(std::ranges::sized_range<std::cmatch>);
static_assert(!std::ranges::borrowed_range<std::cmatch>);
static_assert(std::ranges::viewable_range<std::cmatch>);

static_assert(std::same_as<std::ranges::iterator_t<std::cmatch const>, std::cmatch::const_iterator>);
static_assert(std::ranges::common_range<std::cmatch const>);
static_assert(std::ranges::random_access_range<std::cmatch const>);
static_assert(std::ranges::contiguous_range<std::cmatch const>);
static_assert(!std::ranges::view<std::cmatch const>);
static_assert(std::ranges::sized_range<std::cmatch const>);
static_assert(!std::ranges::borrowed_range<std::cmatch const>);
static_assert(!std::ranges::viewable_range<std::cmatch const>);
