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
//     using iterator_concept = input_iterator_tag;
//     using difference_type = range_difference_t<V>;

#include <concepts>
#include <iterator>
#include <ranges>

#include "test_iterators.h"

// Test `using iterator_concept = input_iterator_tag`
static_assert(std::same_as< typename std::ranges::iterator_t< std::ranges::chunk_view<
                                std::ranges::subrange<cpp17_input_iterator<int*>,
                                                      sentinel_wrapper<cpp17_input_iterator<int*>>>>>::iterator_concept,
                            std::input_iterator_tag>);

// Test `using difference_tyoe = range_difference_t<V>`
static_assert(
    std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                     std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>::
                     difference_type,
                 std::ranges::range_difference_t<
                     std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>);
