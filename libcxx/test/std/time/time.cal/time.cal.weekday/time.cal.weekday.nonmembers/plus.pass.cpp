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

// constexpr weekday operator+(const days& x, const weekday& y) noexcept;
//   Returns: weekday(int{x} + y.count()).
//
// constexpr weekday operator+(const weekday& x, const days& y) noexcept;
//   Returns:
//      weekday{modulo(static_cast<long long>(unsigned{x}) + y.count(), 7)}
//   where modulo(n, 7) computes the remainder of n divided by 7 using Euclidean division.
//   [Note: Given a divisor of 12, Euclidean division truncates towards negative infinity
//   and always produces a remainder in the range of [0, 6].
//   Assuming no overflow in the signed summation, this operation results in a weekday
//   holding a value in the range [0, 6] even if !x.ok(). -end note]
//   [Example: Monday + days{6} == Sunday. -end example]

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
      weekday wd1 = weekday{i} + days{j};
      weekday wd2 = days{j} + weekday{i};
      assert(wd1 == wd2);
      assert((wd1.c_encoding() == euclidian_addition<unsigned, 0, 6>(i, j)));
      assert((wd2.c_encoding() == euclidian_addition<unsigned, 0, 6>(i, j)));
    }

  //  Check the example
  assert(weekday{1} + days{6} == weekday{0});
  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<weekday>() + std::declval<days>());
  ASSERT_SAME_TYPE(weekday, decltype(std::declval<weekday>() + std::declval<days>()));

  ASSERT_NOEXCEPT(std::declval<days>() + std::declval<weekday>());
  ASSERT_SAME_TYPE(weekday, decltype(std::declval<days>() + std::declval<weekday>()));

  test();
  static_assert(test());

  return 0;
}
