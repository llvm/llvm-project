//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class weekday;

// constexpr weekday& operator+=(const days& d) noexcept;
// constexpr weekday& operator-=(const days& d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "../../euclidian.h"

using weekday = std::chrono::weekday;
using days    = std::chrono::days;

constexpr bool test() {
  for (unsigned i = 0; i <= 6; ++i) {
    weekday wd(i);
    assert(((wd += days{3}).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 3)));
    assert(((wd).c_encoding() == euclidian_addition<unsigned, 0, 6>(i, 3)));
  }

  for (unsigned i = 0; i <= 6; ++i) {
    weekday wd(i);
    assert(((wd -= days{4}).c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, 4)));
    assert(((wd).c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, 4)));
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<weekday&>() += std::declval<days&>());
  ASSERT_SAME_TYPE(weekday&, decltype(std::declval<weekday&>() += std::declval<days&>()));

  ASSERT_NOEXCEPT(std::declval<weekday&>() -= std::declval<days&>());
  ASSERT_SAME_TYPE(weekday&, decltype(std::declval<weekday&>() -= std::declval<days&>()));

  test();
  static_assert(test());

  return 0;
}
