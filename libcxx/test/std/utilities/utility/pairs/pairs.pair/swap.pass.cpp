//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// void swap(pair& p);

#include <utility>
#include <cassert>

#include "test_macros.h"

struct S {
    int i;
    TEST_CONSTEXPR_CXX20 S() : i(0) {}
    TEST_CONSTEXPR_CXX20 S(int j) : i(j) {}
    TEST_CONSTEXPR_CXX20 bool operator==(int x) const { return i == x; }
};

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::pair<int, short> P1;
    P1 p1(3, static_cast<short>(4));
    P1 p2(5, static_cast<short>(6));
    p1.swap(p2);
    assert(p1.first == 5);
    assert(p1.second == 6);
    assert(p2.first == 3);
    assert(p2.second == 4);
  }
  {
    typedef std::pair<int, S> P1;
    P1 p1(3, S(4));
    P1 p2(5, S(6));
    p1.swap(p2);
    assert(p1.first == 5);
    assert(p1.second == 6);
    assert(p2.first == 3);
    assert(p2.second == 4);
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
