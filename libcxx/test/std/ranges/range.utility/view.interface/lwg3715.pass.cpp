//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// LWG 3715: `view_interface::empty` is overconstrained

#include <cassert>
#include <ranges>
#include <sstream>

bool test() {
  std::istringstream input("1 2 3 4 5");
  auto i = std::views::istream<int>(input);
  auto r = std::views::counted(i.begin(), 4) | std::views::take(2);
  static_assert(std::ranges::input_range<decltype(r)>);
  static_assert(!std::ranges::forward_range<decltype(r)>);
  static_assert(std::ranges::sized_range<decltype(r)>);
  assert(!r.empty());
  return true;
}

int main(int, char**) {
  test();
  return 0;
}
