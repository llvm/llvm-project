//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class day;

// constexpr day operator-(const day& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const day& x, const day& y) noexcept;
//   Returns: days{int(unsigned{x}) - int(unsigned{y}).

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using day  = std::chrono::day;
using days = std::chrono::days;

constexpr bool test() {
  day dy{12};
  for (unsigned i = 0; i <= 10; ++i) {
    day d1   = dy - days{i};
    days off = dy - day{i};
    assert(static_cast<unsigned>(d1) == 12 - i);
    assert(off.count() == static_cast<int>(12 - i)); // days is signed
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<day>() - std::declval<days>());
  ASSERT_NOEXCEPT(std::declval<day>() - std::declval<day>());

  ASSERT_SAME_TYPE(day, decltype(std::declval<day>() - std::declval<days>()));
  ASSERT_SAME_TYPE(days, decltype(std::declval<day>() - std::declval<day>()));

  test();
  static_assert(test());

  return 0;
}
