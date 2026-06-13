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
//   constexpr synth-three-way-result<T, U>
//     operator<=>(const indirect& lhs, const U& rhs);

#include <cassert>
#include <compare>
#include <limits>
#include <memory>
#include <utility>

#include "test_comparisons.h"

constexpr bool test() {
  {
    const std::indirect<int> i1;
    assert(testOrder(i1, -1, std::strong_ordering::greater));
    assert(testOrder(i1, 0, std::strong_ordering::equal));
    assert(testOrder(i1, 1, std::strong_ordering::less));
  }
  { // A valueless indirect always compares less than the object it is compared with.
    std::indirect<int> i1;
    auto(std::move(i1));
    assert(testOrder(i1, std::numeric_limits<int>::min(), std::strong_ordering::less));
  }

  return true;
}

void test_comparison_throws() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct ComparisonThrows {
    int i = 0;
    bool operator==(ComparisonThrows) const { throw 42; }
    std::strong_ordering operator<=>(ComparisonThrows) const { throw 42; }
  };

  std::indirect<ComparisonThrows> i1(1);
  ComparisonThrows i2(2);
  try {
    (void)(i1 <=> i2);
    assert(false);
  } catch (const int& e) {
    assert(e == 42);
  } catch (...) {
    assert(false);
  }

  auto(std::move(i1));
  assert(testOrder(i1, i2, std::strong_ordering::less));
#endif
}

int main(int, char**) {
  test_comparison_throws();
  test();
  static_assert(test());
  return 0;
}
