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

std::vector<int> vec;
auto input_view = vec | std::ranges::views::to_input;
static_assert(input_view.begin() == input_view.end());
static_assert(input_view.size() == 0);
