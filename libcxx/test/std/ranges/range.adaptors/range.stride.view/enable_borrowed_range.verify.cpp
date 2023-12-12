//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// expected-no-diagnostics

// template<class T>
// inline constexpr bool enable_borrowed_range<stride_view<T>> = false;

#include <ranges>
#include <vector>

// The stride_view is not one of those range adaptors that (under any circumstances)
// is enabled as a borrowable range by default. In other words, we will have to make
// a positive test case explicit.

template <>
inline constexpr bool
    std::ranges::enable_borrowed_range<std::ranges::stride_view<std::views::all_t<std::vector<int>>>> = true;

static_assert(!std::ranges::enable_borrowed_range< std::ranges::stride_view<std::ranges::empty_view<int>>>);
static_assert(std::ranges::enable_borrowed_range< std::ranges::stride_view<std::views::all_t<std::vector<int>>>>);
