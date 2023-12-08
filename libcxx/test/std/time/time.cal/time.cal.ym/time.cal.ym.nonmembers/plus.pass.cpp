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

// constexpr year_month operator+(const year_month& ym, const years& dy) noexcept;
// Returns: (ym.year() + dy) / ym.month().
//
// constexpr year_month operator+(const years& dy, const year_month& ym) noexcept;
// Returns: ym + dy.
//
// constexpr year_month operator+(const year_month& ym, const months& dm) noexcept;
// Returns: A year_month value z such that z.ok() && z - ym == dm is true.
// Complexity: O(1) with respect to the value of dm.
//
// constexpr year_month operator+(const months& dm, const year_month& ym) noexcept;
// Returns: ym + dm.

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year = std::chrono::year;
using years = std::chrono::years;
using month = std::chrono::month;
using months = std::chrono::months;
using year_month = std::chrono::year_month;

// year_month + years
constexpr bool test_ym_plus_y() {
  ASSERT_NOEXCEPT(std::declval<year_month>() + std::declval<years>());
  ASSERT_NOEXCEPT(std::declval<years>() + std::declval<year_month>());

  ASSERT_SAME_TYPE(
      year_month, decltype(std::declval<year_month>() + std::declval<years>()));
  ASSERT_SAME_TYPE(
      year_month, decltype(std::declval<years>() + std::declval<year_month>()));

  year_month ym{year{1234}, std::chrono::January};
  for (int i = 0; i <= 10; ++i) {
    year_month ym1 = ym + years{i};
    year_month ym2 = years{i} + ym;
    assert(static_cast<int>(ym1.year()) == i + 1234);
    assert(static_cast<int>(ym2.year()) == i + 1234);
    assert(ym1.month() == std::chrono::January);
    assert(ym2.month() == std::chrono::January);
    assert(ym1 == ym2);
  }

  return true;
}

// year_month + months
constexpr bool test_ym_plus_m() {
  ASSERT_NOEXCEPT(std::declval<year_month>() + std::declval<months>());
  ASSERT_NOEXCEPT(std::declval<months>() + std::declval<year_month>());

  ASSERT_SAME_TYPE(year_month, decltype(std::declval<year_month>() +
                                        std::declval<months>()));
  ASSERT_SAME_TYPE(year_month, decltype(std::declval<months>() +
                                        std::declval<year_month>()));

  year_month ym{year{1234}, std::chrono::January};
  for (int i = 0; i <= 11; ++i) {
    year_month ym1 = ym + months{i};
    year_month ym2 = months{i} + ym;
    assert(static_cast<int>(ym1.year()) == 1234);
    assert(static_cast<int>(ym2.year()) == 1234);
    assert(ym1.month() == month(1 + i));
    assert(ym2.month() == month(1 + i));
    assert(ym1 == ym2);
  }

  for (int i = 12; i < 23; ++i) {
    year_month ym1 = ym + months{i};
    year_month ym2 = months{i} + ym;
    assert(static_cast<int>(ym1.year()) == 1235);
    assert(static_cast<int>(ym2.year()) == 1235);
    assert(ym1.month() == month(1 + i % 12));
    assert(ym2.month() == month(1 + i % 12));
    assert(ym1 == ym2);
  }

  return true;
}

constexpr bool test() {
  test_ym_plus_y();
  test_ym_plus_m();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
