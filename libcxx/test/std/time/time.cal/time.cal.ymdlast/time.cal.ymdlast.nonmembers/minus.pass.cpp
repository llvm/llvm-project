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
//   operator-(const year_month_day_last& ymdl, const months& dm) noexcept;
//
//   Returns: ymdl + (-dm).
//
// constexpr year_month_day_last
//   operator-(const year_month_day_last& ymdl, const years& dy) noexcept;
//
//   Returns: ymdl + (-dy).

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year                = std::chrono::year;
using month               = std::chrono::month;
using month_day_last      = std::chrono::month_day_last;
using year_month_day_last = std::chrono::year_month_day_last;
using months              = std::chrono::months;
using years               = std::chrono::years;

constexpr bool test() {
  constexpr month December = std::chrono::December;

  { // year_month_day_last - years

    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (int i = 0; i <= 10; ++i) {
      year_month_day_last ym1 = ym - years{i};
      assert(static_cast<int>(ym1.year()) == 1234 - i);
      assert(ym1.month() == December);
    }
  }

  { // year_month_day_last - months

    // TODO test wrapping
    year_month_day_last ym{year{1234}, month_day_last{December}};
    for (unsigned i = 0; i <= 10; ++i) {
      year_month_day_last ym1 = ym - months{i};
      assert(static_cast<int>(ym1.year()) == 1234);
      assert(static_cast<unsigned>(ym1.month()) == 12U - i);
    }
  }

  return true;
}

int main(int, char**) {
  // year_month_day_last - years
  ASSERT_NOEXCEPT(std::declval<year_month_day_last>() - std::declval<years>());
  ASSERT_SAME_TYPE(year_month_day_last, decltype(std::declval<year_month_day_last>() - std::declval<years>()));

  // year_month_day_last - months
  ASSERT_NOEXCEPT(std::declval<year_month_day_last>() - std::declval<months>());
  ASSERT_SAME_TYPE(year_month_day_last, decltype(std::declval<year_month_day_last>() - std::declval<months>()));

  test();
  static_assert(test());

  return 0;
}
