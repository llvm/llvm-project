//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::ranges::stride_view

#include <cassert>
#include <ranges>

bool runtime_test() {
  auto iot    = std::views::iota(1, 22);
  auto str    = std::views::stride(iot, 3);
  auto result = str.size();
  assert(result == 7);
  return true;
}

constexpr bool test() {
  {
    constexpr auto iot = std::views::iota(1, 12);
    constexpr auto str = std::views::stride(iot, 3);
    assert(4 == str.size());
    static_assert(4 == str.size(), "Striding by 3 through a 12 member list has size 4.");
  }
  {
    constexpr auto iot = std::views::iota(1, 22);
    constexpr auto str = std::views::stride(iot, 3);
    assert(7 == str.size());
    static_assert(7 == str.size(), "Striding by 3 through a 12 member list has size 4.");
  }
  return true;
}

int main(int, char**) {
  runtime_test();
  static_assert(test());
  return 0;
}
