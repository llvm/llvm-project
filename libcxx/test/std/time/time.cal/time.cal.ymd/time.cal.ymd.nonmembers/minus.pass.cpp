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

// constexpr year_month_day operator-(const year_month_day& ymd, const years& dy) noexcept;
//    Returns: ymd + (-dy)
//
//  constexpr year_month_day_last operator-(const year_month_day_last& __lhs, const months& __rhs) noexcept;
//    Return: __lhs + (-__rhs)

#include <chrono>
#include <cassert>
#include <ratio>
#include <type_traits>
#include <utility>

#include "test_macros.h"

constexpr bool test_constexpr() {
  std::chrono::year_month_day ym0{std::chrono::year{1234}, std::chrono::January, std::chrono::day{12}};
  std::chrono::year_month_day ym1 = ym0 - std::chrono::years{10};
  return ym1.year() == std::chrono::year{1234 - 10} && ym1.month() == std::chrono::January &&
         ym1.day() == std::chrono::day{12};
}

int main(int, char**) {
  using year           = std::chrono::year;
  using month          = std::chrono::month;
  using day            = std::chrono::day;
  using year_month_day = std::chrono::year_month_day;
  using years          = std::chrono::years;
  using months         = std::chrono::months;
  using decamonths     = std::chrono::duration<int, std::ratio_multiply<std::ratio<10>, months::period>>;
  using decades        = std::chrono::duration<int, std::ratio_multiply<std::ratio<10>, years::period>>;

  ASSERT_NOEXCEPT(std::declval<year_month_day>() - std::declval<years>());
  ASSERT_SAME_TYPE(year_month_day, decltype(std::declval<year_month_day>() - std::declval<years>()));

  constexpr month January = std::chrono::January;

  static_assert(test_constexpr(), "");

  year_month_day ym{year{1234}, January, day{10}};
  for (int i = 0; i <= 10; ++i) {
    year_month_day ym1 = ym - years{i};
    assert(static_cast<int>(ym1.year()) == 1234 - i);
    assert(ym1.month() == January);
    assert(ym1.day() == day{10});
  }

  for (int i = 0; i <= 3; ++i) {
    year_month_day ym1 = ym - decades(i);
    assert(ym1.month() == std::chrono::January);
    assert(ym1.year() == ym.year() - years{i * 10});
    assert(ym1.day() == ym.day());
  }

  for (unsigned int i = 0; i < 5; i++) {
    months added_months = decamonths(i);
    year_month_day ym1  = ym - decamonths(i);
    assert(ym1.month() == ym.month() - decamonths(i));
    int d              = static_cast<int>(static_cast<unsigned>(ym.month())) - 1 - added_months.count();
    int dy             = (d >= 0 ? d : d - 11) / 12;
    year expected_year = ym.year() + years{dy};
    assert(ym1.year() == expected_year);
  }

  return 0;
}
