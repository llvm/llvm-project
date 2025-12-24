//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   General tests for chunk_view. This file does not test anything specifically.

#include <algorithm>
#include <cassert>
#include <ranges>
#include <string_view>

#include "test_range.h"

constexpr bool test() {
  std::string_view str = "Cheese with chicken chunk by chunk on truck with my trick";
  // clang-format off
  auto str2 = str 
            | std::views::chunk(4)
            | std::views::join
            | std::views::chunk(314159)
            | std::views::take(1)
            | std::views::join
            | std::views::lazy_split(' ')
            | std::views::chunk(2)
            | std::views::transform([] (auto&& subview)
              {
                return subview | std::views::join_with(' ');
              })
            | std::views::join_with(' ');
  // clang-format on
  assert(std::ranges::equal(str, str2));
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
