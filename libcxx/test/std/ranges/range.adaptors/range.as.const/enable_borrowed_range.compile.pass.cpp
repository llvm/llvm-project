//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

#include <ranges>
#include <vector>

static_assert(std::ranges::enable_borrowed_range<std::ranges::as_const_view<std::ranges::empty_view<int>>>);
static_assert(std::ranges::enable_borrowed_range<std::ranges::as_const_view<std::views::all_t<std::vector<int>&>>>);
static_assert(!std::ranges::enable_borrowed_range<std::ranges::as_const_view<std::views::all_t<std::vector<int>>>>);
