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

// constexpr bool operator==(const year& x, const year& y) noexcept;
// constexpr strong_ordering operator<=>(const year& x, const year& y) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

constexpr bool test() {
  using year = std::chrono::year;

  // Validate valid value. The range [-32768, 32767] is guaranteed to be allowed.
  assert(testOrderValues<year>(-32768, -32768));
  assert(testOrderValues<year>(-32768, -32767));
  // Largest positive
  assert(testOrderValues<year>(32767, 32766));
  assert(testOrderValues<year>(32767, 32767));

  // Validate some valid values.
  for (int i = 1; i < 10; ++i)
    for (int j = 1; j < 10; ++j)
      assert(testOrderValues<year>(i, j));

  return true;
}

int main(int, char**) {
  using year = std::chrono::year;
  AssertOrderAreNoexcept<year>();
  AssertOrderReturn<std::strong_ordering, year>();

  test();
  static_assert(test());

  return 0;
}
