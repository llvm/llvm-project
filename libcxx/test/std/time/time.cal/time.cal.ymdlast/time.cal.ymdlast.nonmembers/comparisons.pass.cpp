//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>
// class year_month_day_last;

// constexpr bool operator==(const year_month_day_last& x, const year_month_day_last& y) noexcept;
// constexpr bool operator<=>(const year_month_day_last& x, const year_month_day_last& y) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

constexpr bool test() {
  using year                = std::chrono::year;
  using month               = std::chrono::month;
  using month_day_last      = std::chrono::month_day_last;
  using year_month_day_last = std::chrono::year_month_day_last;

  constexpr month January  = std::chrono::January;
  constexpr month February = std::chrono::February;

  assert(testOrder(year_month_day_last{year{1234}, month_day_last{January}},
                   year_month_day_last{year{1234}, month_day_last{January}},
                   std::strong_ordering::equal));

  //  different month
  assert(testOrder(year_month_day_last{year{1234}, month_day_last{January}},
                   year_month_day_last{year{1234}, month_day_last{February}},
                   std::strong_ordering::less));

  //  different year
  assert(testOrder(year_month_day_last{year{1234}, month_day_last{January}},
                   year_month_day_last{year{1235}, month_day_last{January}},
                   std::strong_ordering::less));

  //  different year and month
  assert(testOrder(year_month_day_last{year{1234}, month_day_last{February}},
                   year_month_day_last{year{1235}, month_day_last{January}},
                   std::strong_ordering::less));

  //  same year, different months
  for (unsigned i = 1; i < 12; ++i)
    for (unsigned j = 1; j < 12; ++j)
      assert((testOrder(year_month_day_last{year{1234}, month_day_last{month{i}}},
                        year_month_day_last{year{1234}, month_day_last{month{j}}},
                        i == j  ? std::strong_ordering::equal
                        : i < j ? std::strong_ordering::less
                                : std::strong_ordering::greater)));

  //  same month, different years
  for (int i = 1000; i < 1010; ++i)
    for (int j = 1000; j < 1010; ++j)
      assert((testOrder(year_month_day_last{year{i}, month_day_last{January}},
                        year_month_day_last{year{j}, month_day_last{January}},
                        i == j  ? std::strong_ordering::equal
                        : i < j ? std::strong_ordering::less
                                : std::strong_ordering::greater)));
  return true;
}

int main(int, char**) {
  using year_month_day_last = std::chrono::year_month_day_last;
  AssertOrderAreNoexcept<year_month_day_last>();
  AssertOrderReturn<std::strong_ordering, year_month_day_last>();

  test();
  static_assert(test());

  return 0;
}
