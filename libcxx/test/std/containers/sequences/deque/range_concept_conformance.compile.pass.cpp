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

// deque

#include <deque>

#include <concepts>
#include <ranges>

using range = std::deque<int>;
namespace stdr = std::ranges;

static_assert(std::same_as<stdr::iterator_t<range>, range::iterator>);
static_assert(stdr::common_range<range>);
static_assert(stdr::bidirectional_range<range>);
static_assert(!stdr::view<range>);

static_assert(std::same_as<stdr::iterator_t<range const>, range::const_iterator>);
static_assert(stdr::common_range<range const>);
static_assert(stdr::bidirectional_range<range const>);
static_assert(!stdr::view<range const>);
