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

// constexpr year_month_day& operator+=(const months& m) noexcept;
// constexpr year_month_day& operator-=(const months& m) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year           = std::chrono::year;
using month          = std::chrono::month;
using day            = std::chrono::day;
using year_month_day = std::chrono::year_month_day;
using months         = std::chrono::months;

constexpr bool test() {
  for (unsigned i = 0; i <= 10; ++i) {
    year y{1234};
    day d{23};
    year_month_day ymd(y, month{i}, d);

    ymd += months{2};
    assert(ymd.year() == y);
    assert(ymd.day() == d);
    assert(static_cast<unsigned>((ymd).month()) == i + 2);

    ymd -= months{1};
    assert(ymd.year() == y);
    assert(ymd.day() == d);
    assert(static_cast<unsigned>((ymd).month()) == i + 1);
  }

  { // Validate the ok status when the day is not present in the new month.
    year_month_day ymd{year{2020}, month{3}, day{31}};
    ymd += months{1};
    assert((ymd == year_month_day{year{2020}, month{4}, day{31}}));
    assert(!ymd.ok());

    ymd -= months{1};
    assert((ymd == year_month_day{year{2020}, month{3}, day{31}}));
    assert(ymd.ok());
  }

  { // Validate the ok status when the day becomes present in the new month.
    year_month_day ymd{year{2020}, month{4}, day{31}};
    assert(!ymd.ok());

    ymd += months{1};
    assert((ymd == year_month_day{year{2020}, month{5}, day{31}}));
    assert(ymd.ok());

    ymd -= months{2};
    assert((ymd == year_month_day{year{2020}, month{3}, day{31}}));
    assert(ymd.ok());
  }

  { // Test year wrapping
    year_month_day ymd{year{2020}, month{4}, day{31}};

    ymd += months{12};
    assert((ymd == year_month_day{year{2021}, month{4}, day{31}}));

    ymd -= months{12};
    assert((ymd == year_month_day{year{2020}, month{4}, day{31}}));
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year_month_day&>() += std::declval<months>());
  ASSERT_NOEXCEPT(std::declval<year_month_day&>() -= std::declval<months>());

  ASSERT_SAME_TYPE(year_month_day&, decltype(std::declval<year_month_day&>() += std::declval<months>()));
  ASSERT_SAME_TYPE(year_month_day&, decltype(std::declval<year_month_day&>() -= std::declval<months>()));

  test();
  static_assert(test());

  return 0;
}
