//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year;

// constexpr year operator-(const year& x, const years& y) noexcept;
//   Returns: x + -y.
//
// constexpr years operator-(const year& x, const year& y) noexcept;
//   Returns: If x.ok() == true and y.ok() == true, returns a value m in the range
//   [years{0}, years{11}] satisfying y + m == x.
//   Otherwise the value returned is unspecified.
//   [Example: January - February == years{11}. -end example]

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

using year  = std::chrono::year;
using years = std::chrono::years;

constexpr bool test() {
  year y{1223};
  for (int i = 1100; i <= 1110; ++i) {
    year y1   = y - years{i};
    years ys1 = y - year{i};
    assert(static_cast<int>(y1) == 1223 - i);
    assert(ys1.count() == 1223 - i);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year>() - std::declval<years>());
  ASSERT_SAME_TYPE(year, decltype(std::declval<year>() - std::declval<years>()));

  ASSERT_NOEXCEPT(std::declval<year>() - std::declval<year>());
  ASSERT_SAME_TYPE(years, decltype(std::declval<year>() - std::declval<year>()));

  test();
  static_assert(test());

  return 0;
}
