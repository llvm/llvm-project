//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// LWG 3749: common_iterator should handle integer-class difference types

#include <concepts>
#include <cstdint>
#include <iterator>
#include <ranges>

template <class It, class Se>
concept has_iterator_category =
    requires { typename std::iterator_traits<std::common_iterator<It, Se>>::iterator_category; };

using WideIota = decltype(std::views::iota(std::uint64_t{0}));
using WideIter = std::ranges::iterator_t<WideIota>;
using WideSent = std::ranges::sentinel_t<WideIota>;

constexpr bool is_diff_integral = std::integral<std::iter_difference_t<WideIter>>;
static_assert(has_iterator_category<WideIter, WideSent> == is_diff_integral);
