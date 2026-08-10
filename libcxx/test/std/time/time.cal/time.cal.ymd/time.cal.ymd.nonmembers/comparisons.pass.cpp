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

// constexpr bool operator==(const year_month_day& x, const year_month_day& y) noexcept;
// constexpr strong_ordering operator<=>(const year_month_day& x, const year_month_day& y) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

constexpr bool test() {
  using day            = std::chrono::day;
  using year           = std::chrono::year;
  using month          = std::chrono::month;
  using year_month_day = std::chrono::year_month_day;

  constexpr month January  = std::chrono::January;
  constexpr month February = std::chrono::February;

  assert(testOrder(
      year_month_day{year{1234}, January, day{1}},
      year_month_day{year{1234}, January, day{1}},
      std::strong_ordering::equal));

  // different day
  assert(testOrder(
      year_month_day{year{1234}, January, day{1}},
      year_month_day{year{1234}, January, day{2}},
      std::strong_ordering::less));

  // different month
  assert(testOrder(
      year_month_day{year{1234}, January, day{1}},
      year_month_day{year{1234}, February, day{1}},
      std::strong_ordering::less));

  // different year
  assert(testOrder(
      year_month_day{year{1234}, January, day{1}},
      year_month_day{year{1235}, January, day{1}},
      std::strong_ordering::less));

  // different month and day
  assert(testOrder(
      year_month_day{year{1234}, January, day{2}},
      year_month_day{year{1234}, February, day{1}},
      std::strong_ordering::less));

  // different year and month
  assert(testOrder(
      year_month_day{year{1234}, February, day{1}},
      year_month_day{year{1235}, January, day{1}},
      std::strong_ordering::less));

  // different year and day
  assert(testOrder(
      year_month_day{year{1234}, January, day{2}},
      year_month_day{year{1235}, January, day{1}},
      std::strong_ordering::less));

  // different year, month and day
  assert(testOrder(
      year_month_day{year{1234}, February, day{2}},
      year_month_day{year{1235}, January, day{1}},
      std::strong_ordering::less));

  // same year, different days
  for (unsigned i = 1; i < 28; ++i)
    for (unsigned j = 1; j < 28; ++j)
      assert((testOrder(
          year_month_day{year{1234}, January, day{i}},
          year_month_day{year{1234}, January, day{j}},
          i == j  ? std::strong_ordering::equal
          : i < j ? std::strong_ordering::less
                  : std::strong_ordering::greater)));

  // same year, different months
  for (unsigned i = 1; i < 12; ++i)
    for (unsigned j = 1; j < 12; ++j)
      assert((testOrder(
          year_month_day{year{1234}, month{i}, day{12}},
          year_month_day{year{1234}, month{j}, day{12}},
          i == j  ? std::strong_ordering::equal
          : i < j ? std::strong_ordering::less
                  : std::strong_ordering::greater)));

  // same month, different years
  for (int i = -5; i < 5; ++i)
    for (int j = -5; j < 5; ++j)
      assert((testOrder(
          year_month_day{year{i}, January, day{12}},
          year_month_day{year{j}, January, day{12}},
          i == j  ? std::strong_ordering::equal
          : i < j ? std::strong_ordering::less
                  : std::strong_ordering::greater)));

  return true;
}

int main(int, char**) {
  using year_month_day = std::chrono::year_month_day;
  AssertOrderAreNoexcept<year_month_day>();
  AssertOrderReturn<std::strong_ordering, year_month_day>();

  test();
  static_assert(test());

  return 0;
}
