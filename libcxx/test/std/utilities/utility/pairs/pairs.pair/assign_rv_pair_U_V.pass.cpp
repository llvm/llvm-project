//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(pair<U, V>&& p);

#include <utility>
#include <memory>
#include <cassert>

#include "test_macros.h"
#include "archetypes.h"

struct Derived : ConstexprTestTypes::MoveOnly {
  Derived() = default;
  TEST_CONSTEXPR_CXX20 Derived(ConstexprTestTypes::MoveOnly&&){};
};
struct CountAssign {
  int copied = 0;
  int moved = 0;
  TEST_CONSTEXPR_CXX20 CountAssign() = default;
  TEST_CONSTEXPR_CXX20 CountAssign(const int){};
  TEST_CONSTEXPR_CXX20 CountAssign& operator=(CountAssign const&) {
    ++copied;
    return *this;
  }
  TEST_CONSTEXPR_CXX20 CountAssign& operator=(CountAssign&&) {
    ++moved;
    return *this;
  }
};

struct CopyAssignableInt {
  CopyAssignableInt& operator=(int&) { return *this; }
};

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef std::pair<Derived, short> P1;
    typedef std::pair<ConstexprTestTypes::MoveOnly, long> P2;
    P1 p1(Derived(), static_cast<short>(4));
    P2 p2;
    p2 = std::move(p1);
    assert(p2.second == 4);
  }
  {
    using P = std::pair<int, CountAssign>;
    using T = std::pair<long, CountAssign>;
    T t(42, -42);
    P p(101, 101);
    p = std::move(t);
    assert(p.first == 42);
    assert(p.second.moved == 1);
    assert(p.second.copied == 0);
    assert(t.second.moved == 0);
    assert(t.second.copied == 0);
  }
  { // test const requirement
    using T = std::pair<CopyAssignableInt, CopyAssignableInt>;
    using P = std::pair<int, int>;
    static_assert(!std::is_assignable<T, P&&>::value, "");
    static_assert(!std::is_assignable<P, T&&>::value, "");
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
