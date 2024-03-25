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
  TEST_CONSTEXPR_CXX20 Derived(ConstexprTestTypes::MoveOnly&&) {}
};
struct CountAssign {
  int copied = 0;
  int moved = 0;
  TEST_CONSTEXPR_CXX20 CountAssign() = default;
  TEST_CONSTEXPR_CXX20 CountAssign(const int) {}
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

struct NotAssignable {
  NotAssignable& operator=(NotAssignable const&) = delete;
  NotAssignable& operator=(NotAssignable&&) = delete;
};

struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

struct CopyAssignable {
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&) = delete;
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
    static_assert(!std::is_assignable<T&, P&&>::value, "");
    static_assert(!std::is_assignable<P&, T&&>::value, "");
  }
  {
    // Make sure we can't move-assign from a pair containing a reference
    // if that type isn't copy-constructible (since otherwise we'd be
    // stealing the object through the reference).
    using P1 = std::pair<MoveAssignable, long>;
    using P2 = std::pair<MoveAssignable&, int>;
    static_assert(!std::is_assignable<P1&, P2&&>::value, "");

    // ... but this should work since we're going to steal out of the
    // incoming rvalue reference.
    using P3 = std::pair<MoveAssignable, long>;
    using P4 = std::pair<MoveAssignable&&, int>;
    static_assert(std::is_assignable<P3&, P4&&>::value, "");
  }
  {
    // We assign through the reference and don't move out of the incoming ref,
    // so this doesn't work (but would if the type were CopyAssignable).
    {
      using P1 = std::pair<MoveAssignable&, long>;
      using P2 = std::pair<MoveAssignable&, int>;
      static_assert(!std::is_assignable<P1&, P2&&>::value, "");
    }

    // ... works if it's CopyAssignable
    {
      using P1 = std::pair<CopyAssignable&, long>;
      using P2 = std::pair<CopyAssignable&, int>;
      static_assert(std::is_assignable<P1&, P2&&>::value, "");
    }

    // For rvalue-references, we can move-assign if the type is MoveAssignable,
    // or CopyAssignable (since in the worst case the move will decay into a copy).
    {
      using P1 = std::pair<MoveAssignable&&, long>;
      using P2 = std::pair<MoveAssignable&&, int>;
      static_assert(std::is_assignable<P1&, P2&&>::value, "");

      using P3 = std::pair<CopyAssignable&&, long>;
      using P4 = std::pair<CopyAssignable&&, int>;
      static_assert(std::is_assignable<P3&, P4&&>::value, "");
    }

    // In all cases, we can't move-assign if the types are not assignable,
    // since we assign through the reference.
    {
      using P1 = std::pair<NotAssignable&, long>;
      using P2 = std::pair<NotAssignable&, int>;
      static_assert(!std::is_assignable<P1&, P2&&>::value, "");

      using P3 = std::pair<NotAssignable&&, long>;
      using P4 = std::pair<NotAssignable&&, int>;
      static_assert(!std::is_assignable<P3&, P4&&>::value, "");
    }
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
