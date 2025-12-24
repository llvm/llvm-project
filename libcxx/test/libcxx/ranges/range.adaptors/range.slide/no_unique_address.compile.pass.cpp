//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// This test ensures that we use `[[no_unique_address]]` in `slide_view`.

#include <cstddef>
#include <ranges>
#include <string>
#include <type_traits>

#include "test_iterators.h"
#include "test_range.h"

struct random_access_view {
  int* begin() const;
  int* end() const;
};
template <>
inline constexpr bool std::ranges::enable_view<random_access_view> = true;
static_assert(std::ranges::forward_range<random_access_view>);

using SV = std::ranges::slide_view<random_access_view>;
// Expected SV layout:
// [[no_unique_address]] _View __base_                                                // size: 0
// [[no_unique_address]] range_difference_t<_View> __n_                               // size: sizeof(ptrdiff_t)
// [[no_unique_address]] _If<__slide_caches_first, iterator_t<_View>, __empty_cache_> // size: 0, __slide_caches_nothing
// [[no_unique_address]] _If<__slide_caches_last,  iterator_t<_View>, __empty_cache_> // size: 0, __slide_caches_nothing
// size: two adjacent __empty_cache has 1 byte.
static_assert(sizeof(SV) == sizeof(std::ptrdiff_t) * 2);
