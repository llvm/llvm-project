//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// multiset(); // constexpr since C++26

#include <set>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    std::multiset<int> m;
    assert(m.empty());
    assert(m.begin() == m.end());
  }
#if TEST_STD_VER >= 11
  {
    std::multiset<int, std::less<int>, min_allocator<int>> m;
    assert(m.empty());
    assert(m.begin() == m.end());
  }
  {
    typedef explicit_allocator<int> A;
    {
      std::multiset<int, std::less<int>, A> m;
      assert(m.empty());
      assert(m.begin() == m.end());
    }
    {
      A a;
      std::multiset<int, std::less<int>, A> m(a);
      assert(m.empty());
      assert(m.begin() == m.end());
    }
  }
  {
    std::multiset<int> m = {};
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
  return 0;
}
