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

// constexpr year_month& operator+=(const years& d) noexcept;
// constexpr year_month& operator-=(const years& d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using month      = std::chrono::month;
using year       = std::chrono::year;
using years      = std::chrono::years;
using year_month = std::chrono::year_month;

constexpr bool test() {
  for (int i = 1000; i <= 1010; ++i) {
    month m{2};
    year_month ym(year{i}, m);
    assert(static_cast<int>((ym += years{2}).year()) == i + 2);
    assert(ym.month() == m);
    assert(static_cast<int>((ym).year()) == i + 2);
    assert(ym.month() == m);
    assert(static_cast<int>((ym -= years{1}).year()) == i + 1);
    assert(ym.month() == m);
    assert(static_cast<int>((ym).year()) == i + 1);
    assert(ym.month() == m);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year_month&>() += std::declval<years>());
  ASSERT_SAME_TYPE(year_month&, decltype(std::declval<year_month&>() += std::declval<years>()));

  ASSERT_NOEXCEPT(std::declval<year_month&>() -= std::declval<years>());
  ASSERT_SAME_TYPE(year_month&, decltype(std::declval<year_month&>() -= std::declval<years>()));

  test();
  static_assert(test());

  return 0;
}
