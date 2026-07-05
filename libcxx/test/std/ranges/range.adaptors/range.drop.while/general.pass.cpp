//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Some basic examples of how drop_while_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <algorithm>
#include <cassert>
#include <ranges>
#include <string_view>

int main(int, char**) {
  using namespace std::string_view_literals;
  std::string_view source = "  \t   \t   \t   hello there"sv;
  auto is_invisible       = [](const auto x) { return x == ' ' || x == '\t'; };
  auto skip_ws            = std::views::drop_while(source, is_invisible);
  assert(std::ranges::equal(skip_ws, "hello there"sv));

  return 0;
}
