//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range:
//     class outer_iterator;

//     using outer_iterator::iterator_concept = input_iterator_tag;
//     using outer_iterator::difference_type = range_difference_t<V>;
//     class outer_iterator::value_type;

#include <concepts>
#include <iterator>
#include <ranges>

#include "../types.h"

using OuterIterator = std::ranges::iterator_t<std::ranges::chunk_view<input_span<int>>>;

static_assert(std::same_as<typename OuterIterator::iterator_concept, std::input_iterator_tag>);
static_assert(std::same_as<typename OuterIterator::difference_type, std::ranges::range_difference_t<input_span<int>>>);
static_assert(std::same_as<typename OuterIterator::value_type, std::iter_value_t<OuterIterator>>);
static_assert(std::ranges::input_range<typename OuterIterator::value_type>);
