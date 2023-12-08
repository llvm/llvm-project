//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year;

// constexpr year& operator+=(const years& d) noexcept;
// constexpr year& operator-=(const years& d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year  = std::chrono::year;
using years = std::chrono::years;

constexpr bool test() {
  for (int i = 10000; i <= 10020; ++i) {
    year yr(i);
    assert(static_cast<int>(yr += years{10}) == i + 10);
    assert(static_cast<int>(yr) == i + 10);
    assert(static_cast<int>(yr -= years{9}) == i + 1);
    assert(static_cast<int>(yr) == i + 1);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<year&>() += std::declval<years>());
  ASSERT_NOEXCEPT(std::declval<year&>() -= std::declval<years>());

  ASSERT_SAME_TYPE(year&, decltype(std::declval<year&>() += std::declval<years>()));
  ASSERT_SAME_TYPE(year&, decltype(std::declval<year&>() -= std::declval<years>()));

  test();
  static_assert(test());

  return 0;
}
