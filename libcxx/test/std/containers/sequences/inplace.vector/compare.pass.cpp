//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <vector>

// friend bool operator==(const inplace_vector& lhs, const inplace_vector& rhs);
// synthesized operator!=
// constexpr auto operator<=>(const inplace_vector& x,
//                            const inplace_vector& y);  // synth-three-way-result
// synthesized operator<, operator>, operator<=, operator>=

#include <inplace_vector>
#include <cassert>

#include "test_comparisons.h"

constexpr bool test() {
  {
    const std::inplace_vector<int, 0> c1, c2;
    assert(testComparisons(c1, c2, true, false));
  }
  {
    const std::inplace_vector<int, 10> c1, c2;
    assert(testComparisons(c1, c2, true, false));
  }
  {
    const std::inplace_vector<int, 10> c1(1, 1), c2(1, 2);
    assert(testComparisons(c1, c2, false, true));
  }
  {
    const std::inplace_vector<int, 10> c1, c2(1, 2);
    assert(testComparisons(c1, c2, false, true));
  }
  {
    const std::inplace_vector<int, 10> c1{1, 2, 1};
    const std::inplace_vector<int, 10> c2{1, 2, 2};
    assert(testComparisons(c1, c2, false, true));
  }
  {
    const std::inplace_vector<int, 10> c1{3, 2, 3};
    const std::inplace_vector<int, 10> c2{3, 1, 3};

    assert(testComparisons(c1, c2, false, false));
  }
  {
    const std::inplace_vector<int, 10> c1{1, 2};
    const std::inplace_vector<int, 10> c2{1, 2, 0};
    assert(testComparisons(c1, c2, false, true));
  }
  {
    const std::inplace_vector<int, 10> c1{1, 2, 0};
    const std::inplace_vector<int, 10> c2{3};
    assert(testComparisons(c1, c2, false, true));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 0> c1, c2;
    assert(testComparisons(c1, c2, true, false));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1, c2;
    assert(testComparisons(c1, c2, true, false));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1{LessAndEqComp(1)};
    const std::inplace_vector<LessAndEqComp, 10> c2{LessAndEqComp(1)};
    assert(testComparisons(c1, c2, true, false));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1{LessAndEqComp(1)};
    const std::inplace_vector<LessAndEqComp, 10> c2{LessAndEqComp(2)};
    assert(testComparisons(c1, c2, false, true));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1;
    const std::inplace_vector<LessAndEqComp, 10> c2(1, LessAndEqComp(2));
    assert(testComparisons(c1, c2, false, true));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1{LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(2)};
    const std::inplace_vector<LessAndEqComp, 10> c2{LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(1)};
    assert(testComparisons(c1, c2, false, false));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1{LessAndEqComp(3), LessAndEqComp(3), LessAndEqComp(3)};
    const std::inplace_vector<LessAndEqComp, 10> c2{LessAndEqComp(3), LessAndEqComp(2), LessAndEqComp(3)};
    assert(testComparisons(c1, c2, false, false));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1{LessAndEqComp(1), LessAndEqComp(2)};
    const std::inplace_vector<LessAndEqComp, 10> c2{LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(0)};
    assert(testComparisons(c1, c2, false, true));
  }
  if !consteval {
    const std::inplace_vector<LessAndEqComp, 10> c1{LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(0)};
    const std::inplace_vector<LessAndEqComp, 10> c2{LessAndEqComp(3)};
    assert(testComparisons(c1, c2, false, true));
  }
  {
    using V = std::inplace_vector<int, 0>;
    assert((V() == V()));
    assert(!(V() != V()));
    assert(!(V() < V()));
    assert((V() <= V()));
    assert(!(V() > V()));
    assert((V() >= V()));
  }
  {
    using V = std::inplace_vector<int, 10>;
    assert((V() == V()));
    assert(!(V() != V()));
    assert(!(V() < V()));
    assert((V() <= V()));
    assert(!(V() > V()));
    assert((V() >= V()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
