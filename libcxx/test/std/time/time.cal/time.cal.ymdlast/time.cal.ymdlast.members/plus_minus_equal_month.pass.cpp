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

// constexpr year_month_day_last& operator+=(const months& m) noexcept;
// constexpr year_month_day_last& operator-=(const months& m) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year                = std::chrono::year;
using month               = std::chrono::month;
using month_day_last      = std::chrono::month_day_last;
using year_month_day_last = std::chrono::year_month_day_last;
using months              = std::chrono::months;

constexpr bool test() {
  for (unsigned i = 0; i <= 10; ++i) {
    year y{1234};
    month_day_last mdl{month{i}};
    year_month_day_last ymdl(y, mdl);
    assert(static_cast<unsigned>((ymdl += months{2}).month()) == i + 2);
    assert(ymdl.year() == y);
    assert(static_cast<unsigned>((ymdl).month()) == i + 2);
    assert(ymdl.year() == y);
    assert(static_cast<unsigned>((ymdl -= months{1}).month()) == i + 1);
    assert(ymdl.year() == y);
    assert(static_cast<unsigned>((ymdl).month()) == i + 1);
    assert(ymdl.year() == y);
  }

  { // Test year wrapping
    year_month_day_last ymdl{year{2020}, month_day_last{month{4}}};

    ymdl += months{12};
    assert((ymdl == year_month_day_last{year{2021}, month_day_last{month{4}}}));

    ymdl -= months{12};
    assert((ymdl == year_month_day_last{year{2020}, month_day_last{month{4}}}));
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year_month_day_last&>() += std::declval<months>());
  ASSERT_NOEXCEPT(std::declval<year_month_day_last&>() -= std::declval<months>());

  ASSERT_SAME_TYPE(year_month_day_last&, decltype(std::declval<year_month_day_last&>() += std::declval<months>()));
  ASSERT_SAME_TYPE(year_month_day_last&, decltype(std::declval<year_month_day_last&>() -= std::declval<months>()));

  test();
  static_assert(test());

  return 0;
}
