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
#include "test_allocator.h"

constexpr bool test() {
  { // Comparing indirects compares their held objects.
    const std::indirect<int> i1;
    const std::indirect<int> i2;
    assert(testEquality(i1, i2, true));
    static_assert(noexcept(i1 == i2));
    static_assert(noexcept(i1 != i2));
  }
  { // Indirects can be compared even if they have different allocator types.
    const std::indirect<int> i1;
    const std::indirect<int, other_allocator<int>> i2;
    assert(testEquality(i1, i2, true));
    static_assert(noexcept(i1 == i2));
    static_assert(noexcept(i1 != i2));
  }
  { // Valueless indirects always compare equal to each other and not equal to ones that hold values.
    std::indirect<int> i1;
    std::indirect<int> i2;
    assert(testEquality(i1, i2, true));
    auto(std::move(i1));
    assert(testEquality(i1, i2, false));
    auto(std::move(i2));
    assert(testEquality(i1, i2, true));
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
  std::indirect<ComparisonThrows> i2(2);
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
  auto(std::move(i2));
  assert(testEquality(i1, i2, true));
#endif
}

int main(int, char**) {
  test_comparison_throws();
  test();
  static_assert(test());
  return 0;
}
