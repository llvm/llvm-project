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

// constexpr weekday operator-(const weekday& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const weekday& x, const weekday& y) noexcept;
// Returns: If x.ok() == true and y.ok() == true, returns a value d in the range
//    [days{0}, days{6}] satisfying y + d == x.
// Otherwise the value returned is unspecified.
// [Example: Sunday - Monday == days{6}. -end example]

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "../../euclidian.h"

using weekday = std::chrono::weekday;
using days    = std::chrono::days;

constexpr bool test() {
  for (unsigned i = 0; i <= 6; ++i)
    for (unsigned j = 0; j <= 6; ++j) {
      weekday wd = weekday{i} - days{j};
      assert(wd + days{j} == weekday{i});
      assert((wd.c_encoding() == euclidian_subtraction<unsigned, 0, 6>(i, j)));
    }

  for (unsigned i = 0; i <= 6; ++i)
    for (unsigned j = 0; j <= 6; ++j) {
      days d = weekday{j} - weekday{i};
      assert(weekday{i} + d == weekday{j});
    }

  //  Check the example
  assert(weekday{0} - weekday{1} == days{6});

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<weekday>() - std::declval<days>());
  ASSERT_SAME_TYPE(weekday, decltype(std::declval<weekday>() - std::declval<days>()));

  ASSERT_NOEXCEPT(std::declval<weekday>() - std::declval<weekday>());
  ASSERT_SAME_TYPE(days, decltype(std::declval<weekday>() - std::declval<weekday>()));

  test();
  static_assert(test());

  return 0;
}
