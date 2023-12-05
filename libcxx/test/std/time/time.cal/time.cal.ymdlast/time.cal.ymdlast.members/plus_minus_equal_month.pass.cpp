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
    year_month_day_last ym(y, mdl);
    assert(static_cast<unsigned>((ym += months{2}).month()) == i + 2);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym).month()) == i + 2);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym -= months{1}).month()) == i + 1);
    assert(ym.year() == y);
    assert(static_cast<unsigned>((ym).month()) == i + 1);
    assert(ym.year() == y);
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
