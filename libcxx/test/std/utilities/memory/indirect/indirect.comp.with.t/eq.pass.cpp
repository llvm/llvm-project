//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// template <class T, class Allocator = std::allocator<T>> class indirect;

// template<class U>
//   constexpr bool operator==(const indirect& lhs, const U& rhs) noexcept(noexcept(*lhs == rhs));

#include <cassert>
#include <memory>
#include <utility>

#include "test_comparisons.h"

constexpr bool test() {
  {
    const std::indirect<int> i1(1);
    const int i2 = 2;
    assert(testEquality(i1, 2, false));
    static_assert(noexcept(i1 == i2));
    static_assert(noexcept(i2 == i1));
    static_assert(noexcept(i1 != i2));
    static_assert(noexcept(i2 != i1));
  }
  { // A valueless indirect always compares false.
    std::indirect<int> i1;
    const int i2 = 0;
    assert(testEquality(i1, i2, true));
    auto(std::move(i1));
    assert(testEquality(i1, i2, false));
  }

  return true;
}

void test_comparison_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct ComparisonThrows {
    int i = 0;
    bool operator==(ComparisonThrows) const { throw 42; }
  };

  std::indirect<ComparisonThrows> i1(1);
  ComparisonThrows i2(2);
  static_assert(!noexcept(i1 == i2));
  static_assert(!noexcept(i1 != i2));

  try {
    (void)(i1 == i2);
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }

  auto(std::move(i1));
  assert(testEquality(i1, i2, false));
#endif
}

int main(int, char**) {
  test_comparison_throws();
  test();
  static_assert(test());
  return 0;
}
