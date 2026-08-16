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
//     inner_iterator(inner_iterator&&) = default;
//     inner_iterator& operator=(inner_iterator&&) = default;

#include <concepts>
#include <ranges>

#include "test_iterators.h"

// Test `inner_iterator() = delete`
static_assert(
    !std::default_initializable<std::ranges::iterator_t< std::ranges::range_reference_t<std::ranges::chunk_view<
        std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>>);

// Test `inner_iterator(const inner_iterator&) = delete`
static_assert(!std::copy_constructible<std::ranges::iterator_t< std::ranges::range_reference_t<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>>);

// Test `inner_iterator& operator=(const inner_iterator&) = delete`
static_assert(!std::assignable_from<
              std::ranges::iterator_t< std::ranges::range_reference_t<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>&,
              const std::ranges::iterator_t< std::ranges::range_reference_t< std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>&>);

// Test `inner_iterator(inner_iterator&&) = default`
static_assert(std::move_constructible<std::ranges::iterator_t< std::ranges::range_reference_t<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>>);

// Test `inner_iterator& operator=(inner_iterator&) = default`
static_assert(std::assignable_from<
              std::ranges::iterator_t< std::ranges::range_reference_t<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>&,
              std::ranges::iterator_t< std::ranges::range_reference_t<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>>>);
