//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   General tests for slide_view. This file does not test anything specifically.

#include <algorithm>
#include <cassert>
#include <ranges>
#include <string_view>

constexpr bool test() {
  std::string_view str = "amanaplanacanalpanama"; // A man, a plan, a canal, Panama!
  auto str2            = str | std::views::slide(4) | std::views::join;
  auto str3            = str | std::views::reverse | std::views::slide(4) | std::views::join;
  assert(std::ranges::equal(str2, str3));
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
