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

#include "test_iterators.h"

static_assert(std::same_as< typename std::ranges::iterator_t< std::ranges::chunk_view<
                                std::ranges::subrange<cpp17_input_iterator<int*>,
                                                      sentinel_wrapper<cpp17_input_iterator<int*>>>>>::iterator_concept,
                            std::input_iterator_tag>);
static_assert(
    std::same_as<typename std::ranges::iterator_t< std::ranges::chunk_view<
                     std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>::
                     difference_type,
                 std::ranges::range_difference_t<
                     std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>);
static_assert(std::same_as<
              typename std::ranges::iterator_t<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>::
                  value_type,
              std::iter_value_t<std::ranges::iterator_t<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>>);
static_assert(
    std::ranges::input_range< typename std::ranges::iterator_t<std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>::value_type>);
