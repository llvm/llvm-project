//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <utility>

// template <class T1, class T2> struct pair
// constexpr const pair& operator=(pair&& p) const;

#include <cassert>
#include <utility>

#include "test_macros.h"
#include "copy_move_types.h"

// Constraints:
// is_assignable<const first_type&, first_type> is true and
// is_assignable<const second_type&, second_type> is true.

// clang-format off
static_assert(std::is_assignable_v<const std::pair<int&&, int&&>&,
                                   std::pair<int&&, int&&>&&>);
static_assert(!std::is_assignable_v<const std::pair<int, int>&,
                                    std::pair<int, int>&&>);
static_assert(!std::is_assignable_v<const std::pair<int, int&&>&,
                                    std::pair<int, int&&>&&>);
static_assert(!std::is_assignable_v<const std::pair<int&&, int>&,
                                    std::pair<int&&, int>&&>);

static_assert(std::is_assignable_v<const std::pair<ConstMoveAssign, ConstMoveAssign>&,
                                   std::pair<ConstMoveAssign, ConstMoveAssign>&&>);
static_assert(!std::is_assignable_v<const std::pair<MoveAssign, MoveAssign>&,
                                   std::pair<MoveAssign, MoveAssign>&&>);

// clang-format on

constexpr bool test() {
  // reference types
  {
    int i1    = 1;
    int i2    = 2;
    double d1 = 3.0;
    double d2 = 5.0;
    std::pair<int&&, double&&> p1{std::move(i1), std::move(d1)};
    const std::pair<int&&, double&&> p2{std::move(i2), std::move(d2)};
    p2 = std::move(p1);
    assert(p2.first == 1);
    assert(p2.second == 3.0);
  }

  // user defined const move assignment
  {
    std::pair<ConstMoveAssign, ConstMoveAssign> p1{1, 2};
    const std::pair<ConstMoveAssign, ConstMoveAssign> p2{3, 4};
    p2 = std::move(p1);
    assert(p2.first.val == 1);
    assert(p2.second.val == 2);
  }

  // The correct assignment operator of the underlying type is used
  {
    std::pair<TracedAssignment, const TracedAssignment> t1{};
    const std::pair<TracedAssignment, const TracedAssignment> t2{};
    t2 = std::move(t1);
    assert(t2.first.constMoveAssign == 1);
    assert(t2.second.constCopyAssign == 1);
  }

  return true;
}

int main(int, const char**) {
  test();
// gcc cannot have mutable member in constant expression
#if !defined(TEST_COMPILER_GCC)
  static_assert(test());
#endif
}
