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

#include "test.h"
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
    // Test with ranges that are sized_range true.
    constexpr auto iot_twelve = std::views::iota(1, 12);
    static_assert(std::ranges::sized_range<decltype(iot_twelve)>);
    constexpr auto str_iot_twelve = std::views::stride(iot_twelve, 3);
    static_assert(std::ranges::sized_range<decltype(str_iot_twelve)>);
    assert(4 == str_iot_twelve.size());
    static_assert(4 == str_iot_twelve.size(), "Striding by 3 through a 12 member list has size 4.");

    constexpr auto iot_twenty_two = std::views::iota(1, 22);
    static_assert(std::ranges::sized_range<decltype(iot_twenty_two)>);
    constexpr auto str_iot_twenty_two = std::views::stride(iot_twenty_two, 3);
    static_assert(std::ranges::sized_range<decltype(str_iot_twenty_two)>);

    assert(7 == str_iot_twenty_two.size());
    static_assert(7 == str_iot_twenty_two.size(), "Striding by 3 through a 22 member list has size 4.");
  }

  {
    // Test with ranges that are not sized_range true.
    constexpr auto unsized_range = UnsizedBasicRange();
    static_assert(!std::ranges::sized_range<decltype(unsized_range)>);
    constexpr auto str_unsized = std::views::stride(unsized_range, 3);
    static_assert(!std::ranges::sized_range<decltype(str_unsized)>);
  }
  return true;
}

int main(int, char**) {
  runtime_test();
  static_assert(test());
  return 0;
}
