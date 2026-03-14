//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// duration

// template<class Rep1, class Period1, class Rep2, class Period2>
//     requires ThreeWayComparable<typename CT::rep>
//   constexpr auto operator<=>(const duration<Rep1, Period1>& lhs,
//                              const duration<Rep2, Period2>& rhs);

#include <cassert>
#include <chrono>
#include <ratio>

#include "test_comparisons.h"

constexpr bool test() {
  {
    std::chrono::seconds s1(3);
    std::chrono::seconds s2(3);
    assert((s1 <=> s2) == std::strong_ordering::equal);
    assert(testOrder(s1, s2, std::strong_ordering::equal));
  }
  {
    std::chrono::seconds s1(3);
    std::chrono::seconds s2(4);
    assert((s1 <=> s2) == std::strong_ordering::less);
    assert(testOrder(s1, s2, std::strong_ordering::less));
  }
  {
    std::chrono::milliseconds s1(3);
    std::chrono::microseconds s2(3000);
    assert((s1 <=> s2) == std::strong_ordering::equal);
    assert(testOrder(s1, s2, std::strong_ordering::equal));
  }
  {
    std::chrono::milliseconds s1(3);
    std::chrono::microseconds s2(4000);
    assert((s1 <=> s2) == std::strong_ordering::less);
    assert(testOrder(s1, s2, std::strong_ordering::less));
  }
  {
    std::chrono::duration<int, std::ratio<2, 3>> s1(9);
    std::chrono::duration<int, std::ratio<3, 5>> s2(10);
    assert((s1 <=> s2) == std::strong_ordering::equal);
    assert(testOrder(s1, s2, std::strong_ordering::equal));
  }
  {
    std::chrono::duration<int, std::ratio<2, 3>> s1(10);
    std::chrono::duration<int, std::ratio<3, 5>> s2(9);
    assert((s1 <=> s2) == std::strong_ordering::greater);
    assert(testOrder(s1, s2, std::strong_ordering::greater));
  }
  {
    std::chrono::duration<int, std::ratio<2, 3>> s1(9);
    std::chrono::duration<double, std::ratio<3, 5>> s2(10.1);
    assert((s1 <=> s2) == std::strong_ordering::less);
    assert(testOrder(s1, s2, std::strong_ordering::less));
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
