//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

//  template<input_range V>
//    requires view<V>
//  class as_input_view : public view_interface<as_input_view<V>>

//   template<class R>
//     as_input_view(R&&) -> as_input_view<views::all_t<R>>;

#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

static_assert(std::same_as<decltype(std::ranges::as_input_view(std::vector<int>{})),
                           std::ranges::as_input_view<std::views::all_t<std::vector<int>>>>);

static_assert(std::same_as<decltype(std::ranges::as_input_view(std::declval<std::vector<int>&>())),
                           std::ranges::as_input_view<std::views::all_t<std::vector<int>&>>>);

static_assert(std::same_as<decltype(std::ranges::as_input_view(std::ranges::empty_view<int>{})),
                           std::ranges::as_input_view<std::ranges::empty_view<int>>>);
