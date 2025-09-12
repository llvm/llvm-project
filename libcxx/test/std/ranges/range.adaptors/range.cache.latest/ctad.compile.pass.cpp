//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class cache_latest_view

//   template<class R>
//     cache_latest_view(R&&) -> cache_latest_view<views::all_t<R>>;

#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

static_assert(std::same_as<decltype(std::ranges::cache_latest_view(std::vector<int>{})),
                           std::ranges::cache_latest_view<std::views::all_t<std::vector<int>>>>);

static_assert(std::same_as<decltype(std::ranges::cache_latest_view(std::declval<std::vector<int>&>())),
                           std::ranges::cache_latest_view<std::views::all_t<std::vector<int>&>>>);

static_assert(std::same_as<decltype(std::ranges::cache_latest_view(std::ranges::empty_view<int>{})),
                           std::ranges::cache_latest_view<std::ranges::empty_view<int>>>);