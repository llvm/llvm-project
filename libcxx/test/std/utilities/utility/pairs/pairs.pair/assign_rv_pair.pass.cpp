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

// pair& operator=(pair&& p);

#include <utility>
#include <memory>
#include <cassert>

#include "test_macros.h"
#include "archetypes.h"

struct CountAssign {
  int copied = 0;
  int moved = 0;
  TEST_CONSTEXPR_CXX20 CountAssign() = default;
  TEST_CONSTEXPR_CXX20 CountAssign& operator=(CountAssign const&) {
    ++copied;
    return *this;
  }
  TEST_CONSTEXPR_CXX20 CountAssign& operator=(CountAssign&&) {
    ++moved;
    return *this;
  }
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
    typedef std::pair<ConstexprTestTypes::MoveOnly, int> P;
    P p1(3, 4);
    P p2;
    p2 = std::move(p1);
    assert(p2.first.value == 3);
    assert(p2.second == 4);
  }
  {
    using P = std::pair<int&, int&&>;
    int x = 42;
    int y = 101;
    int x2 = -1;
    int y2 = 300;
    P p1(x, std::move(y));
    P p2(x2, std::move(y2));
    p1 = std::move(p2);
    assert(p1.first == x2);
    assert(p1.second == y2);
  }
  {
    using P = std::pair<int, ConstexprTestTypes::DefaultOnly>;
    static_assert(!std::is_move_assignable<P>::value, "");
  }
  {
    // The move decays to the copy constructor
    using P = std::pair<CountAssign, ConstexprTestTypes::CopyOnly>;
    static_assert(std::is_move_assignable<P>::value, "");
    P p;
    P p2;
    p = std::move(p2);
    assert(p.first.moved == 0);
    assert(p.first.copied == 1);
    assert(p2.first.moved == 0);
    assert(p2.first.copied == 0);
  }
  {
    using P = std::pair<CountAssign, ConstexprTestTypes::MoveOnly>;
    static_assert(std::is_move_assignable<P>::value, "");
    P p;
    P p2;
    p = std::move(p2);
    assert(p.first.moved == 1);
    assert(p.first.copied == 0);
    assert(p2.first.moved == 0);
    assert(p2.first.copied == 0);
  }
  {
    using P1 = std::pair<int, NotAssignable>;
    using P2 = std::pair<NotAssignable, int>;
    using P3 = std::pair<NotAssignable, NotAssignable>;
    static_assert(!std::is_move_assignable<P1>::value, "");
    static_assert(!std::is_move_assignable<P2>::value, "");
    static_assert(!std::is_move_assignable<P3>::value, "");
  }
  {
    // We assign through the reference and don't move out of the incoming ref,
    // so this doesn't work (but would if the type were CopyAssignable).
    using P1 = std::pair<MoveAssignable&, int>;
    static_assert(!std::is_move_assignable<P1>::value, "");

    // ... works if it's CopyAssignable
    using P2 = std::pair<CopyAssignable&, int>;
    static_assert(std::is_move_assignable<P2>::value, "");

    // For rvalue-references, we can move-assign if the type is MoveAssignable
    // or CopyAssignable (since in the worst case the move will decay into a copy).
    using P3 = std::pair<MoveAssignable&&, int>;
    using P4 = std::pair<CopyAssignable&&, int>;
    static_assert(std::is_move_assignable<P3>::value, "");
    static_assert(std::is_move_assignable<P4>::value, "");

    // In all cases, we can't move-assign if the types are not assignable,
    // since we assign through the reference.
    using P5 = std::pair<NotAssignable&, int>;
    using P6 = std::pair<NotAssignable&&, int>;
    static_assert(!std::is_move_assignable<P5>::value, "");
    static_assert(!std::is_move_assignable<P6>::value, "");
  }

#if TEST_STD_VER >= 20 && defined(_LIBCPP_ABI_PAIR_TRIVIALLY_COPYABLE)
  static_assert(std::is_trivially_move_assignable<std::pair<int, int>>::value, "");
#else
  static_assert(!std::is_trivially_move_assignable<std::pair<int, int>>::value, "");
#endif

  static_assert(!std::is_trivially_move_assignable<std::pair<CountAssign, int>>::value, "");
  static_assert(!std::is_trivially_move_assignable<std::pair<int, CountAssign>>::value, "");
  static_assert(!std::is_trivially_move_assignable<std::pair<CountAssign, CountAssign>>::value, "");

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
