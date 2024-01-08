//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization

// Check LWG-3698: `regex_iterator` and `join_view` don't work together very well

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <regex>
#include <string_view>

int main(int, char**) {
  char const text[] = "Hello";
  std::regex regex{"[a-z]"};

  auto lower =
      std::ranges::subrange(
          std::cregex_iterator(std::ranges::begin(text), std::ranges::end(text), regex), std::cregex_iterator{}) |
      std::views::join | std::views::transform([](auto const& sm) { return std::string_view(sm.first, sm.second); });

  assert(std::ranges::equal(lower, std::to_array<std::string_view>({"e", "l", "l", "o"})));

  return 0;
}
