//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_weekday;

// constexpr year_month_weekday operator-(const year_month_weekday& ymwd, const months& dm) noexcept;
//   Returns: ymwd + (-dm).
//
// constexpr year_month_weekday operator-(const year_month_weekday& ymwd, const years& dy) noexcept;
//   Returns: ymwd + (-dy).

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

using year               = std::chrono::year;
using month              = std::chrono::month;
using weekday            = std::chrono::weekday;
using weekday_indexed    = std::chrono::weekday_indexed;
using year_month_weekday = std::chrono::year_month_weekday;
using years              = std::chrono::years;
using months             = std::chrono::months;

constexpr bool test() {
  constexpr month November  = std::chrono::November;
  constexpr weekday Tuesday = std::chrono::Tuesday;

  { // year_month_weekday - years
    year_month_weekday ymwd{year{1234}, November, weekday_indexed{Tuesday, 1}};
    for (int i = 0; i <= 10; ++i) {
      year_month_weekday ymwd1 = ymwd - years{i};
      assert(static_cast<int>(ymwd1.year()) == 1234 - i);
      assert(ymwd1.month() == November);
      assert(ymwd1.weekday() == Tuesday);
      assert(ymwd1.index() == 1);
    }
  }

  { // year_month_weekday - months
    year_month_weekday ymwd{year{1234}, November, weekday_indexed{Tuesday, 2}};
    for (unsigned i = 1; i <= 10; ++i) {
      year_month_weekday ymwd1 = ymwd - months{i};
      assert(ymwd1.year() == year{1234});
      assert(ymwd1.month() == month{11 - i});
      assert(ymwd1.weekday() == Tuesday);
      assert(ymwd1.index() == 2);
    }
    // Test the year wraps around.
    for (unsigned i = 12; i <= 15; ++i) {
      year_month_weekday ymwd1 = ymwd - months{i};
      assert(ymwd1.year() == year{1233});
      assert(ymwd1.month() == month{11 - i + 12});
      assert(ymwd1.weekday() == Tuesday);
      assert(ymwd1.index() == 2);
    }
  }

  return true;
}

int main(int, char**) {
  // year_month_weekday - years
  ASSERT_NOEXCEPT(std::declval<year_month_weekday>() - std::declval<years>());
  ASSERT_SAME_TYPE(year_month_weekday, decltype(std::declval<year_month_weekday>() - std::declval<years>()));

  // year_month_weekday - months
  ASSERT_NOEXCEPT(std::declval<year_month_weekday>() - std::declval<months>());
  ASSERT_SAME_TYPE(year_month_weekday, decltype(std::declval<year_month_weekday>() - std::declval<months>()));

  test();
  static_assert(test());

  return 0;
}
