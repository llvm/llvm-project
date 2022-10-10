//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Range concept conformance tests for take_view.

#include <ranges>
#include <cassert>
#include <concepts>

#include "test_iterators.h"
#include "test_range.h"

static_assert(std::ranges::input_range<std::ranges::take_view<test_view<cpp20_input_iterator>>>);
static_assert(std::ranges::forward_range<std::ranges::take_view<test_view<forward_iterator>>>);
static_assert(std::ranges::bidirectional_range<std::ranges::take_view<test_view<bidirectional_iterator>>>);
static_assert(std::ranges::random_access_range<std::ranges::take_view<test_view<random_access_iterator>>>);
static_assert(std::ranges::contiguous_range<std::ranges::take_view<test_view<contiguous_iterator>>>);

