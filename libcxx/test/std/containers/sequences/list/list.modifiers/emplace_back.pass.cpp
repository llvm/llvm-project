//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <list>

// template <class... Args> reference emplace_back(Args&&... args); // constexpr since C++26
// return type is 'reference' in C++17; 'void' before

#include <list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

class A {
  int i_;
  double d_;

  A(const A&);
  A& operator=(const A&);

public:
  TEST_CONSTEXPR_CXX20 A(int i, double d) : i_(i), d_(d) {}

  TEST_CONSTEXPR int geti() const { return i_; }
  TEST_CONSTEXPR double getd() const { return d_; }
};

TEST_CONSTEXPR_CXX26 bool test() {
  {
    std::list<A> c;
#if TEST_STD_VER > 14
    A& r1 = c.emplace_back(2, 3.5);
    assert(c.size() == 1);
    assert(&r1 == &c.back());
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    A& r2 = c.emplace_back(3, 4.5);
    assert(c.size() == 2);
    assert(&r2 == &c.back());
#else
    c.emplace_back(2, 3.5);
    assert(c.size() == 1);
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    c.emplace_back(3, 4.5);
    assert(c.size() == 2);
#endif
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    assert(c.back().geti() == 3);
    assert(c.back().getd() == 4.5);
  }
  {
    std::list<A, min_allocator<A>> c;
#if TEST_STD_VER > 14
    A& r1 = c.emplace_back(2, 3.5);
    assert(c.size() == 1);
    assert(&r1 == &c.back());
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    A& r2 = c.emplace_back(3, 4.5);
    assert(c.size() == 2);
    assert(&r2 == &c.back());
#else
    c.emplace_back(2, 3.5);
    assert(c.size() == 1);
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    c.emplace_back(3, 4.5);
    assert(c.size() == 2);
#endif
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    assert(c.back().geti() == 3);
    assert(c.back().getd() == 4.5);
  }

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
