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

// constexpr year_month_weekday_last& operator+=(const months& m) noexcept;
// constexpr year_month_weekday_last& operator-=(const months& m) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year                    = std::chrono::year;
using month                   = std::chrono::month;
using weekday                 = std::chrono::weekday;
using weekday_last            = std::chrono::weekday_last;
using year_month_weekday_last = std::chrono::year_month_weekday_last;
using months                  = std::chrono::months;

constexpr bool test() {
  constexpr weekday Tuesday = std::chrono::Tuesday;

  for (unsigned i = 0; i <= 10; ++i) {
    year y{1234};
    year_month_weekday_last ymwdl(y, month{i}, weekday_last{Tuesday});

    assert(static_cast<unsigned>((ymwdl += months{2}).month()) == i + 2);
    assert(ymwdl.year() == y);
    assert(ymwdl.weekday() == Tuesday);

    assert(static_cast<unsigned>((ymwdl).month()) == i + 2);
    assert(ymwdl.year() == y);
    assert(ymwdl.weekday() == Tuesday);

    assert(static_cast<unsigned>((ymwdl -= months{1}).month()) == i + 1);
    assert(ymwdl.year() == y);
    assert(ymwdl.weekday() == Tuesday);

    assert(static_cast<unsigned>((ymwdl).month()) == i + 1);
    assert(ymwdl.year() == y);
    assert(ymwdl.weekday() == Tuesday);
  }

  { // Test year wrapping
    year_month_weekday_last ymwdl{year{2020}, month{4}, weekday_last{Tuesday}};

    ymwdl += months{12};
    assert((ymwdl == year_month_weekday_last{year{2021}, month{4}, weekday_last{Tuesday}}));

    ymwdl -= months{12};
    assert((ymwdl == year_month_weekday_last{year{2020}, month{4}, weekday_last{Tuesday}}));
  }
  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year_month_weekday_last&>() += std::declval<months>());
  ASSERT_NOEXCEPT(std::declval<year_month_weekday_last&>() -= std::declval<months>());

  ASSERT_SAME_TYPE(
      year_month_weekday_last&, decltype(std::declval<year_month_weekday_last&>() += std::declval<months>()));
  ASSERT_SAME_TYPE(
      year_month_weekday_last&, decltype(std::declval<year_month_weekday_last&>() -= std::declval<months>()));

  test();
  static_assert(test());

  return 0;
}
