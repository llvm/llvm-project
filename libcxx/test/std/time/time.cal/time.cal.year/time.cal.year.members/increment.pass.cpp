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

//  constexpr year& operator++() noexcept;
//  constexpr year operator++(int) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using year = std::chrono::year;

constexpr bool test() {
  for (int i = 11000; i <= 11020; ++i) {
    year yr(i);
    assert(static_cast<int>(++yr) == i + 1);
    assert(static_cast<int>(yr++) == i + 1);
    assert(static_cast<int>(yr) == i + 2);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(++(std::declval<year&>()));
  ASSERT_NOEXCEPT((std::declval<year&>())++);

  ASSERT_SAME_TYPE(year, decltype(std::declval<year&>()++));
  ASSERT_SAME_TYPE(year&, decltype(++std::declval<year&>()));

  test();
  static_assert(test());

  return 0;
}
