//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::chunk

#include <ranges>

#include <algorithm>
#include <cassert>
#include <string_view>

#include "test_range.h"

constexpr bool test() {
  auto str = std::string_view("Cheese the chicken chunk by chunk on truck by truck");
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