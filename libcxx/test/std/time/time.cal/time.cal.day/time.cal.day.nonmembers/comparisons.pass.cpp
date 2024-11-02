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

// constexpr bool operator==(const day& x, const day& y) noexcept;
//   Returns: unsigned{x} == unsigned{y}.
// constexpr strong_ordering operator<=>(const day& x, const day& y) noexcept;
//   Returns: unsigned{x} <=> unsigned{y}.

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

constexpr bool test() {
  using day = std::chrono::day;

  // Validate invalid values. The range [0, 255] is guaranteed to be allowed.
  assert(testOrderValues<day>(0U, 0U));
  assert(testOrderValues<day>(0U, 1U));
  assert(testOrderValues<day>(254U, 255U));
  assert(testOrderValues<day>(255U, 255U));

  // Validate some valid values.
  for (unsigned i = 1; i < 10; ++i)
    for (unsigned j = 1; j < 10; ++j)
      assert(testOrderValues<day>(i, j));

  return true;
}

int main(int, char**) {
  using day = std::chrono::day;
  AssertOrderAreNoexcept<day>();
  AssertOrderReturn<std::strong_ordering, day>();

  test();
  static_assert(test());

  return 0;
}
