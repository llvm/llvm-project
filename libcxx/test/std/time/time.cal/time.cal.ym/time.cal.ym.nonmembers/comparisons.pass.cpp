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

// constexpr bool operator==(const year_month& x, const year_month& y) noexcept;
// constexpr strong_order operator<=>(const year_month& x, const year_month& y) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

constexpr bool test() {
  using year       = std::chrono::year;
  using month      = std::chrono::month;
  using year_month = std::chrono::year_month;

  assert(testOrder(
      year_month{year{1234}, std::chrono::January},
      year_month{year{1234}, std::chrono::January},
      std::strong_ordering::equal));

  assert(testOrder(
      year_month{year{1234}, std::chrono::January},
      year_month{year{1234}, std::chrono::February},
      std::strong_ordering::less));

  assert(testOrder(
      year_month{year{1234}, std::chrono::January},
      year_month{year{1235}, std::chrono::January},
      std::strong_ordering::less));

  //  same year, different months
  for (unsigned i = 1; i < 12; ++i)
    for (unsigned j = 1; j < 12; ++j)
      assert((testOrder(
          year_month{year{1234}, month{i}},
          year_month{year{1234}, month{j}},
          i == j  ? std::strong_ordering::equal
          : i < j ? std::strong_ordering::less
                  : std::strong_ordering::greater)));

  //  same month, different years
  for (int i = -5; i < 5; ++i)
    for (int j = -5; j < 5; ++j)
      assert((testOrder(
          year_month{year{i}, std::chrono::January},
          year_month{year{j}, std::chrono::January},
          i == j  ? std::strong_ordering::equal
          : i < j ? std::strong_ordering::less
                  : std::strong_ordering::greater)));

  return true;
}

int main(int, char**) {
  using year_month = std::chrono::year_month;
  AssertOrderAreNoexcept<year_month>();
  AssertOrderReturn<std::strong_ordering, year_month>();

  test();
  static_assert(test());

  return 0;
}
