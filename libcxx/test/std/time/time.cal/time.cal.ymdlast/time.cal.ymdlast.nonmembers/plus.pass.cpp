//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_day_last;

// constexpr year_month_day_last
//   operator+(const year_month_day_last& ymdl, const months& dm) noexcept;
//
//   Returns: (ymdl.year() / ymdl.month() + dm) / last.
//
// constexpr year_month_day_last
//   operator+(const months& dm, const year_month_day_last& ymdl) noexcept;
//
//   Returns: ymdl + dm.
//
//
// constexpr year_month_day_last
//   operator+(const year_month_day_last& ymdl, const years& dy) noexcept;
//
//   Returns: {ymdl.year()+dy, ymdl.month_day_last()}.
//
// constexpr year_month_day_last
//   operator+(const years& dy, const year_month_day_last& ymdl) noexcept;
//
//   Returns: ymdl + dy

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

using year                = std::chrono::year;
using month               = std::chrono::month;
using month_day_last      = std::chrono::month_day_last;
using year_month_day_last = std::chrono::year_month_day_last;
using months              = std::chrono::months;
using years               = std::chrono::years;

constexpr bool test() {
  constexpr month January = std::chrono::January;

  { // year_month_day_last + months
    year_month_day_last ym{year{1234}, month_day_last{January}};
    for (int i = 0; i <= 10; ++i) {
      year_month_day_last ymdl1 = ym + months{i};
      year_month_day_last ymdl2 = months{i} + ym;
      assert(static_cast<int>(ymdl1.year()) == 1234);
      assert(static_cast<int>(ymdl2.year()) == 1234);
      assert(ymdl1.month() == month(1 + i));
      assert(ymdl2.month() == month(1 + i));
      assert(ymdl1 == ymdl2);
    }
    // Test the year wraps around.
    for (int i = 12; i <= 15; ++i) {
      year_month_day_last ymdl1 = ym + months{i};
      year_month_day_last ymdl2 = months{i} + ym;
      assert(static_cast<int>(ymdl1.year()) == 1235);
      assert(static_cast<int>(ymdl2.year()) == 1235);
      assert(ymdl1.month() == month(1 + i - 12));
      assert(ymdl2.month() == month(1 + i - 12));
      assert(ymdl1 == ymdl2);
    }
  }

  { // year_month_day_last + years
    year_month_day_last ym{year{1234}, month_day_last{January}};
    for (int i = 0; i <= 10; ++i) {
      year_month_day_last ymdl1 = ym + years{i};
      year_month_day_last ymdl2 = years{i} + ym;
      assert(static_cast<int>(ymdl1.year()) == i + 1234);
      assert(static_cast<int>(ymdl2.year()) == i + 1234);
      assert(ymdl1.month() == std::chrono::January);
      assert(ymdl2.month() == std::chrono::January);
      assert(ymdl1 == ymdl2);
    }
  }

  return true;
}

int main(int, char**) {
  // year_month_day_last + months
  ASSERT_NOEXCEPT(std::declval<year_month_day_last>() + std::declval<months>());
  ASSERT_NOEXCEPT(std::declval<months>() + std::declval<year_month_day_last>());

  ASSERT_SAME_TYPE(year_month_day_last, decltype(std::declval<year_month_day_last>() + std::declval<months>()));
  ASSERT_SAME_TYPE(year_month_day_last, decltype(std::declval<months>() + std::declval<year_month_day_last>()));

  // year_month_day_last + years
  ASSERT_NOEXCEPT(std::declval<year_month_day_last>() + std::declval<years>());
  ASSERT_NOEXCEPT(std::declval<years>() + std::declval<year_month_day_last>());

  ASSERT_SAME_TYPE(year_month_day_last, decltype(std::declval<year_month_day_last>() + std::declval<years>()));
  ASSERT_SAME_TYPE(year_month_day_last, decltype(std::declval<years>() + std::declval<year_month_day_last>()));

  test();
  static_assert(test());

  return 0;
}
