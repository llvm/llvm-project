//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: std-at-least-c++26

#include <ranges>
#include <vector>

auto vec        = std::vector<int>{2, 3, 4};
auto input_view = vec | std::ranges::views::to_input;
static_assert(std::ranges::input_range<decltype(input_view)>);
static_assert(std::ranges::view<decltype(input_view)>);
static_assert(std::ranges::sized_range<decltype(input_view)>);
static_assert(std::ranges::borrowed_range<decltype(input_view)>);
