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
//     class inner_iterator;

//     using inner_iterator::iterator_concept = input_iterator_tag;
//     using inner_iterator::difference_type = range_difference_t<V>;
//     using inner_iterator::value_type = range_value_t<V>;

#include <concepts>
#include <iterator>
#include <ranges>

#include "../types.h"

using InnerIterator = std::ranges::iterator_t<std::ranges::range_reference_t<std::ranges::chunk_view<input_span<int>>>>;

static_assert(std::same_as<typename InnerIterator::iterator_concept, std::input_iterator_tag>);
static_assert(std::same_as<typename InnerIterator::difference_type, std::ranges::range_difference_t<input_span<int>>>);
static_assert(std::same_as<typename InnerIterator::value_type, std::ranges::range_value_t<input_span<int>>>);
