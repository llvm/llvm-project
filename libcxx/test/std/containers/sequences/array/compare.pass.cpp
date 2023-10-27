//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <class T, size_t N>
//   bool operator==(const array<T,N>& x, const array<T,N>& y);    // constexpr in C++20
// template <class T, size_t N>
//   bool operator!=(const array<T,N>& x, const array<T,N>& y);    // removed in C++20
// template <class T, size_t N>
//   bool operator<(const array<T,N>& x, const array<T,N>& y);     // removed in C++20
// template <class T, size_t N>
//   bool operator>(const array<T,N>& x, const array<T,N>& y);     // removed in C++20
// template <class T, size_t N>
//   bool operator<=(const array<T,N>& x, const array<T,N>& y);    // removed in C++20
// template <class T, size_t N>
//   bool operator>=(const array<T,N>& x, const array<T,N>& y);    // removed in C++20

#include <array>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  // Arrays where the elements support all comparison operators
  AssertComparisonsReturnBool<std::array<int, 3> >();
  {
    typedef std::array<int, 3> C;
    const C c1 = {1, 2, 3};
    const C c2 = {1, 2, 3};
    const C c3 = {3, 2, 1};
    const C c4 = {1, 2, 1};
    assert(testComparisons(c1, c2, true, false));
    assert(testComparisons(c1, c3, false, true));
    assert(testComparisons(c1, c4, false, false));
  }
  // Empty array
  {
    typedef std::array<int, 0> C;
    const C c1 = {};
    const C c2 = {};
    assert(testComparisons(c1, c2, true, false));
  }
  // Arrays where the elements support only less and equality comparisons
  AssertComparisonsReturnBool<std::array<LessAndEqComp, 3> >();
  {
    typedef std::array<LessAndEqComp, 3> C;
    const C c1 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(3)};
    const C c2 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(3)};
    const C c3 = {LessAndEqComp(3), LessAndEqComp(2), LessAndEqComp(1)};
    const C c4 = {LessAndEqComp(1), LessAndEqComp(2), LessAndEqComp(1)};
    assert(testComparisons(c1, c2, true, false));
    assert(testComparisons(c1, c3, false, true));
    assert(testComparisons(c1, c4, false, false));
  }
  // Empty array where the elements support only less and equality comparisons
  {
    typedef std::array<LessAndEqComp, 0> C;
    const C c1 = {};
    const C c2 = {};
    assert(testComparisons(c1, c2, true, false));
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests());
#endif
  return 0;
}
