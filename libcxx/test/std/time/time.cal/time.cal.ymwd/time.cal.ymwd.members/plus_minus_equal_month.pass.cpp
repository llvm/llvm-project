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

// constexpr year_month_weekday& operator+=(const months& m) noexcept;
// constexpr year_month_weekday& operator-=(const months& m) noexcept;

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
using months             = std::chrono::months;

constexpr bool test() {
  constexpr weekday Tuesday = std::chrono::Tuesday;

  for (unsigned i = 0; i <= 10; ++i) {
    year y{1234};
    year_month_weekday ymwd(y, month{i}, weekday_indexed{Tuesday, 2});

    assert(static_cast<unsigned>((ymwd += months{2}).month()) == i + 2);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd).month()) == i + 2);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd -= months{1}).month()) == i + 1);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);

    assert(static_cast<unsigned>((ymwd).month()) == i + 1);
    assert(ymwd.year() == y);
    assert(ymwd.weekday() == Tuesday);
    assert(ymwd.index() == 2);
  }

  { // Test year wrapping
    year_month_weekday ymwd{year{2020}, month{4}, weekday_indexed{Tuesday, 2}};

    ymwd += months{12};
    assert((ymwd == year_month_weekday{year{2021}, month{4}, weekday_indexed{Tuesday, 2}}));

    ymwd -= months{12};
    assert((ymwd == year_month_weekday{year{2020}, month{4}, weekday_indexed{Tuesday, 2}}));
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year_month_weekday&>() += std::declval<months>());
  ASSERT_SAME_TYPE(year_month_weekday&, decltype(std::declval<year_month_weekday&>() += std::declval<months>()));

  ASSERT_NOEXCEPT(std::declval<year_month_weekday&>() -= std::declval<months>());
  ASSERT_SAME_TYPE(year_month_weekday&, decltype(std::declval<year_month_weekday&>() -= std::declval<months>()));

  test();
  static_assert(test());

  return 0;
}
