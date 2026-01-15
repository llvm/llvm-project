//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// This test ensures that we use `[[no_unique_address]]` in `chunk_view`.

#include <cstddef>
#include <ranges>
#include <string>
#include <type_traits>

#include "test_iterators.h"
#include "test_range.h"

// When V models only input_range

struct input_view {
  cpp20_input_iterator<int*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<int*>> end() const;
};
template <>
inline constexpr bool std::ranges::enable_view<input_view> = true;
static_assert(std::ranges::input_range<input_view> && !std::ranges::forward_range<input_view>);

using CV1 = std::ranges::chunk_view<input_view>;
// Expected CV1 (when V models only input_range) layout:
// [[no_unique_address]] _View __base_                                         // size: 0
// [[no_unique_address]] range_difference_t<_View> __n_                        // size: sizeof(ptrdiff_t)
// [[no_unique_address]] range_difference_t<_View> __remainder_                // size: sizeof(ptrdiff_t)
// [[no_unique_address]] __non_propagating_cache<iterator_t<_View>> __current_ // size: sizeof(__non_propagating_cache<cpp20_input_iterator<int*>>), align: std::ptrdiff_t
static_assert(alignof(std::ranges::__non_propagating_cache<cpp20_input_iterator<int*>>) == alignof(std::ptrdiff_t));
static_assert(sizeof(CV1) == /*sizeof(__base_) == 0 + */ sizeof(std::ptrdiff_t) * 2 +
                                 sizeof(std::ranges::__non_propagating_cache<cpp20_input_iterator<int*>>));

// When V models forward_range

struct forward_view {
  int* begin() const;
  int* end() const;
};
template <>
inline constexpr bool std::ranges::enable_view<forward_view> = true;
static_assert(std::ranges::forward_range<forward_view>);

using CV2 = std::ranges::chunk_view<forward_view>;
// Expected CV2 (when V models forward_range) layout:
// [[no_unique_address]] _View __base_             // size: 0
// [[no_unique_address]] range_difference_t<_View> // size: sizeof(ptrdiff_t)
static_assert(sizeof(CV2) == /*sizeof(__base_) == 0 + */ sizeof(std::ptrdiff_t));
