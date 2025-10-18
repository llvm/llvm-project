//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// multimap(); // constexpr since C++26

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26
bool test() {
  {
    std::multimap<int, double> m;
    assert(m.empty());
    assert(m.begin() == m.end());
  }
#if TEST_STD_VER >= 11
  {
    std::multimap<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> m;
    assert(m.empty());
    assert(m.begin() == m.end());
  }
  {
    typedef explicit_allocator<std::pair<const int, double>> A;
    {
      std::multimap<int, double, std::less<int>, A> m;
      assert(m.empty());
      assert(m.begin() == m.end());
    }
    {
      A a;
      std::multimap<int, double, std::less<int>, A> m(a);
      assert(m.empty());
      assert(m.begin() == m.end());
    }
  }
  {
    std::multimap<int, double> m = {};
    assert(m.empty());
    assert(m.begin() == m.end());
  }
#endif

  return true;
}
int main(int, char**) {
  test();

#if TEST_STD_VER >= 26
  static_assert(test());
#endif
}
