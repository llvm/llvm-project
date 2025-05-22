//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_day;

// constexpr year_month_day& operator+=(const years& d) noexcept;
// constexpr year_month_day& operator-=(const years& d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year           = std::chrono::year;
using month          = std::chrono::month;
using day            = std::chrono::day;
using year_month_day = std::chrono::year_month_day;
using years          = std::chrono::years;

constexpr bool test() {
  for (int i = 1000; i <= 1010; ++i) {
    month m{2};
    day d{23};
    year_month_day ym(year{i}, m, d);
    assert(static_cast<int>((ym += years{2}).year()) == i + 2);
    assert(ym.month() == m);
    assert(ym.day() == d);
    assert(static_cast<int>((ym).year()) == i + 2);
    assert(ym.month() == m);
    assert(ym.day() == d);
    assert(static_cast<int>((ym -= years{1}).year()) == i + 1);
    assert(ym.month() == m);
    assert(ym.day() == d);
    assert(static_cast<int>((ym).year()) == i + 1);
    assert(ym.month() == m);
    assert(ym.day() == d);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year_month_day&>() += std::declval<years>());
  ASSERT_NOEXCEPT(std::declval<year_month_day&>() -= std::declval<years>());

  ASSERT_SAME_TYPE(year_month_day&, decltype(std::declval<year_month_day&>() += std::declval<years>()));
  ASSERT_SAME_TYPE(year_month_day&, decltype(std::declval<year_month_day&>() -= std::declval<years>()));

  test();
  static_assert(test());

  return 0;
}
