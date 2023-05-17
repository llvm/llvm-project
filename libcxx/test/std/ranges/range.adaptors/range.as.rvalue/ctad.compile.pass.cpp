//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <ranges>
#include <vector>
#include <utility>

static_assert(std::is_same_v<decltype(std::ranges::as_rvalue_view(std::vector<int>{})),
                             std::ranges::as_rvalue_view<std::views::all_t<std::vector<int>>>>);

static_assert(std::is_same_v<decltype(std::ranges::as_rvalue_view(std::declval<std::vector<int>&>())),
                             std::ranges::as_rvalue_view<std::views::all_t<std::vector<int>&>>>);

static_assert(std::is_same_v<decltype(std::ranges::as_rvalue_view(std::ranges::empty_view<int>{})),
                             std::ranges::as_rvalue_view<std::ranges::empty_view<int>>>);
