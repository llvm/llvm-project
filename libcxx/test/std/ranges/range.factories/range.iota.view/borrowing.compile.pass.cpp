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

// template<class W, class Bound>
//   inline constexpr bool enable_borrowed_range<iota_view<W, Bound>> = true;

#include <ranges>

static_assert(std::ranges::borrowed_range<std::ranges::iota_view<int, int>>);
static_assert(std::ranges::borrowed_range<std::ranges::iota_view<int, std::unreachable_sentinel_t>>);
static_assert(std::ranges::borrowed_range<std::ranges::iota_view<int*, int*>>);
static_assert(std::ranges::borrowed_range<std::ranges::iota_view<int*, std::unreachable_sentinel_t>>);
