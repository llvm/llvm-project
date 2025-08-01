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

// constexpr year operator+() const noexcept;
// constexpr year operator-() const noexcept;

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

using year = std::chrono::year;

constexpr bool test() {
  for (int i = 10000; i <= 10020; ++i) {
    year yr(i);
    assert(static_cast<int>(+yr) == i);
    assert(static_cast<int>(-yr) == -i);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(+std::declval<year>());
  ASSERT_NOEXCEPT(-std::declval<year>());

  ASSERT_SAME_TYPE(year, decltype(+std::declval<year>()));
  ASSERT_SAME_TYPE(year, decltype(-std::declval<year>()));

  test();
  static_assert(test());

  return 0;
}
