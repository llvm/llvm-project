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

// constexpr day& operator+=(const days& d) noexcept;
// constexpr day& operator-=(const days& d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using day  = std::chrono::day;
using days = std::chrono::days;

constexpr bool test() {
  for (unsigned i = 0; i <= 10; ++i) {
    day d(i);
    assert(static_cast<unsigned>(d += days{22}) == i + 22);
    assert(static_cast<unsigned>(d) == i + 22);
    assert(static_cast<unsigned>(d -= days{12}) == i + 10);
    assert(static_cast<unsigned>(d) == i + 10);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<day&>() += std::declval<days>());
  ASSERT_NOEXCEPT(std::declval<day&>() -= std::declval<days>());

  ASSERT_SAME_TYPE(day&, decltype(std::declval<day&>() += std::declval<days>()));
  ASSERT_SAME_TYPE(day&, decltype(std::declval<day&>() -= std::declval<days>()));

  test();
  static_assert(test());

  return 0;
}
