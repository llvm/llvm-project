//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month;

// constexpr year_month operator-(const year_month& ym, const years& dy) noexcept;
// Returns: ym + -dy.
//
// constexpr year_month operator-(const year_month& ym, const months& dm) noexcept;
// Returns: ym + -dm.
//
// constexpr months operator-(const year_month& x, const year_month& y) noexcept;
// Returns: x.year() - y.year() + months{static_cast<int>(unsigned{x.month()}) -
//                                       static_cast<int>(unsigned{y.month()})}

#include <chrono>
#include <cassert>
#include <ratio>
#include <type_traits>
#include <utility>

#include "test_macros.h"

using year       = std::chrono::year;
using years      = std::chrono::years;
using month      = std::chrono::month;
using months     = std::chrono::months;
using year_month = std::chrono::year_month;
using decamonths = std::chrono::duration<int, std::ratio_multiply<std::ratio<10>, months::period>>;
using decades    = std::chrono::duration<int, std::ratio_multiply<std::ratio<10>, years::period>>;

constexpr bool test() {
  { // year_month - years

    year_month ym{year{1234}, std::chrono::January};
    for (int i = 0; i <= 10; ++i) {
      year_month ym1 = ym - years{i};
      assert(static_cast<int>(ym1.year()) == 1234 - i);
      assert(ym1.month() == std::chrono::January);
    }

    for (int i = 0; i <= 3; ++i) {
      year_month ym1 = ym - decades(i);
      assert(ym1.month() == std::chrono::January);
      assert(ym1.year() == ym.year() - years{i * 10});
    }
  }

  { // year_month - months

    year_month ym{year{1234}, std::chrono::November};
    for (int i = 0; i <= 10; ++i) {
      year_month ym1 = ym - months{i};
      assert(static_cast<int>(ym1.year()) == 1234);
      assert(ym1.month() == month(11 - i));
    }
    // Test the year wraps around.
    for (int i = 12; i <= 15; ++i) {
      year_month ym1 = ym - months{i};
      assert(static_cast<int>(ym1.year()) == 1233);
      assert(ym1.month() == month(11 - i + 12));
    }

    for (unsigned int i = 0; i < 5; i++) {
      months added_months = decamonths(i);
      year_month ym1      = ym - decamonths(i);
      assert(ym1.month() == ym.month() - decamonths(i));
      int d              = static_cast<int>(static_cast<unsigned>(ym.month())) - 1 - added_months.count();
      int dy             = (d >= 0 ? d : d - 11) / 12;
      year expected_year = ym.year() + years{dy};
      assert(ym1.year() == expected_year);
    }
  }

  { // year_month - year_month

    //  Same year
    year y{2345};
    for (int i = 1; i <= 12; ++i)
      for (int j = 1; j <= 12; ++j) {
        months diff = year_month{y, month(i)} - year_month{y, month(j)};
        assert(diff.count() == i - j);
      }

    // TODO: different year
  }
  return true;
}

int main(int, char**) {
  // year_month - years
  ASSERT_NOEXCEPT(std::declval<year_month>() - std::declval<years>());
  ASSERT_SAME_TYPE(year_month, decltype(std::declval<year_month>() - std::declval<years>()));

  // year_month - months
  ASSERT_NOEXCEPT(std::declval<year_month>() - std::declval<months>());
  ASSERT_SAME_TYPE(year_month, decltype(std::declval<year_month>() - std::declval<months>()));

  // year_month - year_month
  ASSERT_NOEXCEPT(std::declval<year_month>() - std::declval<year_month>());
  ASSERT_SAME_TYPE(months, decltype(std::declval<year_month>() - std::declval<year_month>()));

  test();
  static_assert(test());

  return 0;
}
