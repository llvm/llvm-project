//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class month;

// constexpr bool operator==(const month& x, const month& y) noexcept;
// constexpr strong_ordering operator<=>(const month& x, const month& y) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

constexpr bool test() {
  using month = std::chrono::month;

  // Validate invalid values. The range [0, 255] is guaranteed to be allowed.
  assert(testOrderValues<month>(0U, 0U));
  assert(testOrderValues<month>(0U, 1U));
  assert(testOrderValues<month>(254U, 255U));
  assert(testOrderValues<month>(255U, 255U));

  // Validate some valid values.
  for (unsigned i = 1; i <= 12; ++i)
    for (unsigned j = 1; j <= 12; ++j)
      assert(testOrderValues<month>(i, j));

  return true;
}

int main(int, char**) {
  using month = std::chrono::month;
  AssertOrderAreNoexcept<month>();
  AssertOrderReturn<std::strong_ordering, month>();

  test();
  static_assert(test());

  return 0;
}
