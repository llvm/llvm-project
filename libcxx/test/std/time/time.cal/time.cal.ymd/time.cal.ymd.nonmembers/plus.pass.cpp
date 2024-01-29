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

// constexpr year_month_day operator+(const year_month_day& ymd, const months& dm) noexcept;
//   Returns: (ymd.year() / ymd.month() + dm) / ymd.day().
//
// constexpr year_month_day operator+(const months& dm, const year_month_day& ymd) noexcept;
//   Returns: ymd + dm.
//
//
// constexpr year_month_day operator+(const year_month_day& ymd, const years& dy) noexcept;
//   Returns: (ymd.year() + dy) / ymd.month() / ymd.day().
//
// constexpr year_month_day operator+(const years& dy, const year_month_day& ymd) noexcept;
//   Returns: ym + dm.

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using day            = std::chrono::day;
using year           = std::chrono::year;
using years          = std::chrono::years;
using month          = std::chrono::month;
using months         = std::chrono::months;
using year_month_day = std::chrono::year_month_day;

constexpr bool test() {
  { // year_month_day + months
    year_month_day ym{year{1234}, std::chrono::January, day{12}};
    for (int i = 0; i <= 10; ++i) // TODO test wrap-around
    {
      year_month_day ym1 = ym + months{i};
      year_month_day ym2 = months{i} + ym;
      assert(static_cast<int>(ym1.year()) == 1234);
      assert(static_cast<int>(ym2.year()) == 1234);
      assert(ym1.month() == month(1 + i));
      assert(ym2.month() == month(1 + i));
      assert(ym1.day() == day{12});
      assert(ym2.day() == day{12});
      assert(ym1 == ym2);
    }
  }

  { // year_month_day + years
    year_month_day ym{year{1234}, std::chrono::January, day{12}};
    for (int i = 0; i <= 10; ++i) {
      year_month_day ym1 = ym + years{i};
      year_month_day ym2 = years{i} + ym;
      assert(static_cast<int>(ym1.year()) == i + 1234);
      assert(static_cast<int>(ym2.year()) == i + 1234);
      assert(ym1.month() == std::chrono::January);
      assert(ym2.month() == std::chrono::January);
      assert(ym1.day() == day{12});
      assert(ym2.day() == day{12});
      assert(ym1 == ym2);
    }
  }

  return true;
}

int main(int, char**) {
  // year_month_day + months
  ASSERT_NOEXCEPT(std::declval<year_month_day>() + std::declval<months>());
  ASSERT_NOEXCEPT(std::declval<months>() + std::declval<year_month_day>());

  ASSERT_SAME_TYPE(year_month_day, decltype(std::declval<year_month_day>() + std::declval<months>()));
  ASSERT_SAME_TYPE(year_month_day, decltype(std::declval<months>() + std::declval<year_month_day>()));

  // year_month_day + years
  ASSERT_NOEXCEPT(std::declval<year_month_day>() + std::declval<years>());
  ASSERT_NOEXCEPT(std::declval<years>() + std::declval<year_month_day>());

  ASSERT_SAME_TYPE(year_month_day, decltype(std::declval<year_month_day>() + std::declval<years>()));
  ASSERT_SAME_TYPE(year_month_day, decltype(std::declval<years>() + std::declval<year_month_day>()));

  test();
  static_assert(test());

  return 0;
}
