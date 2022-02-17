//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// ranges::next
// Make sure we're SFINAE-friendly when the template argument constraints are not met.

#include <iterator>

#include <cstddef>
#include <memory>
#include <utility>
#include "test_iterators.h"

template <class ...Args>
concept has_ranges_next = requires (Args&& ...args) {
  { std::ranges::next(std::forward<Args>(args)...) };
};

using It = std::unique_ptr<int>;
static_assert(!has_ranges_next<It>);
static_assert(!has_ranges_next<It, std::ptrdiff_t>);
static_assert(!has_ranges_next<It, It>);
static_assert(!has_ranges_next<It, std::ptrdiff_t, It>);

// Test the test
using It2 = forward_iterator<int*>;
static_assert(has_ranges_next<It2>);
static_assert(has_ranges_next<It2, std::ptrdiff_t>);
static_assert(has_ranges_next<It2, std::ptrdiff_t, It2>);
