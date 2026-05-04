//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// [Example 1:
//   vector<string> vs = {"the", "quick", "brown", "fox"};
//   for (char c : vs | views::join_with('-')) {
//     cout << c;
//   }
//   // The above prints the-quick-brown-fox
// - end example]

#include <ranges>

#include <cassert>
#include <string>
#include <vector>

constexpr bool test() {
  std::vector<std::string> vs = {"the", "quick", "brown", "fox"};
  std::string result;
  for (char c : vs | std::views::join_with('-')) {
    result += c;
  }

  return result == "the-quick-brown-fox";
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
