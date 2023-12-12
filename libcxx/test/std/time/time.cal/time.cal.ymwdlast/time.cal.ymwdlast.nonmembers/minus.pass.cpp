//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_weekday_last;

// constexpr year_month_weekday_last operator-(const year_month_weekday_last& ymwdl, const months& dm) noexcept;
//   Returns: ymwdl + (-dm).
//
// constexpr year_month_weekday_last operator-(const year_month_weekday_last& ymwdl, const years& dy) noexcept;
//   Returns: ymwdl + (-dy).

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year                    = std::chrono::year;
using month                   = std::chrono::month;
using weekday                 = std::chrono::weekday;
using weekday_last            = std::chrono::weekday_last;
using year_month_weekday_last = std::chrono::year_month_weekday_last;
using years                   = std::chrono::years;
using months                  = std::chrono::months;

constexpr bool test() {
  constexpr month October   = std::chrono::October;
  constexpr weekday Tuesday = std::chrono::Tuesday;

  { // year_month_weekday_last - years
    year_month_weekday_last ym{year{1234}, October, weekday_last{Tuesday}};
    for (int i = 0; i <= 10; ++i) {
      year_month_weekday_last ym1 = ym - years{i};
      assert(ym1.year() == year{1234 - i});
      assert(ym1.month() == October);
      assert(ym1.weekday() == Tuesday);
      assert(ym1.weekday_last() == weekday_last{Tuesday});
    }
  }

  { // year_month_weekday_last - months
    year_month_weekday_last ym{year{1234}, October, weekday_last{Tuesday}};
    for (unsigned i = 0; i < 10; ++i) {
      year_month_weekday_last ym1 = ym - months{i};
      assert(ym1.year() == year{1234});
      assert(ym1.month() == month{10 - i});
      assert(ym1.weekday() == Tuesday);
      assert(ym1.weekday_last() == weekday_last{Tuesday});
    }
  }

  return true;
}

int main(int, char**) {
  // year_month_weekday_last - years
  ASSERT_NOEXCEPT(std::declval<year_month_weekday_last>() - std::declval<years>());
  ASSERT_SAME_TYPE(year_month_weekday_last, decltype(std::declval<year_month_weekday_last>() - std::declval<years>()));

  // year_month_weekday_last - months
  ASSERT_NOEXCEPT(std::declval<year_month_weekday_last>() - std::declval<months>());
  ASSERT_SAME_TYPE(year_month_weekday_last, decltype(std::declval<year_month_weekday_last>() - std::declval<months>()));

  test();
  static_assert(test());

  return 0;
}
