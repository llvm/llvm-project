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

// constexpr year_month_weekday operator+(const year_month_weekday& ymd, const months& dm) noexcept;
//   Returns: (ymd.year() / ymd.month() + dm) / ymd.day().
//
// constexpr year_month_weekday operator+(const months& dm, const year_month_weekday& ymd) noexcept;
//   Returns: ymd + dm.
//
//
// constexpr year_month_weekday operator+(const year_month_weekday& ymd, const years& dy) noexcept;
//   Returns: (ymd.year() + dy) / ymd.month() / ymd.day().
//
// constexpr year_month_weekday operator+(const years& dy, const year_month_weekday& ymd) noexcept;
//   Returns: ym + dm.

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year               = std::chrono::year;
using month              = std::chrono::month;
using weekday            = std::chrono::weekday;
using weekday_indexed    = std::chrono::weekday_indexed;
using year_month_weekday = std::chrono::year_month_weekday;
using years              = std::chrono::years;
using months             = std::chrono::months;

constexpr bool test() {
  constexpr weekday Tuesday = std::chrono::Tuesday;
  constexpr month January   = std::chrono::January;

  { // year_month_weekday + months (and switched)
    year_month_weekday ym{year{1234}, January, weekday_indexed{Tuesday, 3}};
    for (int i = 0; i <= 10; ++i) {
      year_month_weekday ymwd1 = ym + months{i};
      year_month_weekday ymwd2 = months{i} + ym;
      assert(static_cast<int>(ymwd1.year()) == 1234);
      assert(static_cast<int>(ymwd2.year()) == 1234);
      assert(ymwd1.month() == month(1 + i));
      assert(ymwd2.month() == month(1 + i));
      assert(ymwd1.weekday() == Tuesday);
      assert(ymwd2.weekday() == Tuesday);
      assert(ymwd1.index() == 3);
      assert(ymwd2.index() == 3);
      assert(ymwd1 == ymwd2);
    }
    // Test the year wraps around.
    for (int i = 12; i <= 15; ++i) {
      year_month_weekday ymwd1 = ym + months{i};
      year_month_weekday ymwd2 = months{i} + ym;
      assert(static_cast<int>(ymwd1.year()) == 1235);
      assert(static_cast<int>(ymwd2.year()) == 1235);
      assert(ymwd1.month() == month(1 + i - 12));
      assert(ymwd2.month() == month(1 + i - 12));
      assert(ymwd1.weekday() == Tuesday);
      assert(ymwd2.weekday() == Tuesday);
      assert(ymwd1.index() == 3);
      assert(ymwd2.index() == 3);
      assert(ymwd1 == ymwd2);
    }
  }

  { // year_month_weekday + years (and switched)
    year_month_weekday ym{year{1234}, std::chrono::January, weekday_indexed{Tuesday, 3}};
    for (int i = 0; i <= 10; ++i) {
      year_month_weekday ymwd1 = ym + years{i};
      year_month_weekday ymwd2 = years{i} + ym;
      assert(static_cast<int>(ymwd1.year()) == i + 1234);
      assert(static_cast<int>(ymwd2.year()) == i + 1234);
      assert(ymwd1.month() == January);
      assert(ymwd2.month() == January);
      assert(ymwd1.weekday() == Tuesday);
      assert(ymwd2.weekday() == Tuesday);
      assert(ymwd1.index() == 3);
      assert(ymwd2.index() == 3);
      assert(ymwd1 == ymwd2);
    }
  }

  return true;
}

int main(int, char**) {
  // year_month_weekday + months (and switched)
  ASSERT_NOEXCEPT(std::declval<year_month_weekday>() + std::declval<months>());
  ASSERT_NOEXCEPT(std::declval<months>() + std::declval<year_month_weekday>());

  ASSERT_SAME_TYPE(year_month_weekday, decltype(std::declval<year_month_weekday>() + std::declval<months>()));
  ASSERT_SAME_TYPE(year_month_weekday, decltype(std::declval<months>() + std::declval<year_month_weekday>()));

  // year_month_weekday + years (and switched)
  ASSERT_NOEXCEPT(std::declval<year_month_weekday>() + std::declval<years>());
  ASSERT_NOEXCEPT(std::declval<years>() + std::declval<year_month_weekday>());

  ASSERT_SAME_TYPE(year_month_weekday, decltype(std::declval<year_month_weekday>() + std::declval<years>()));
  ASSERT_SAME_TYPE(year_month_weekday, decltype(std::declval<years>() + std::declval<year_month_weekday>()));

  test();
  static_assert(test());

  return 0;
}
