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
// template<class U1, class U2> constexpr
// const pair& operator=(const pair<U1, U2>& p) const;

#include <cassert>
#include <utility>

#include "test_macros.h"
#include "copy_move_types.h"

// Constraints:
// is_assignable_v<const first_type&, const U1&> is true, and
// is_assignable_v<const second_type&, const U2&> is true.

// clang-format off
static_assert( std::is_assignable_v<const std::pair<int&, int&>&,
                                    const std::pair<long&, long&>&>);
static_assert(!std::is_assignable_v<const std::pair<int, int>&,
                                    const std::pair<long, long>&>);
static_assert(!std::is_assignable_v<const std::pair<int, int&>&,
                                    const std::pair<long, long&>&>);
static_assert(!std::is_assignable_v<const std::pair<int&, int>&,
                                    const std::pair<long&, long>&>);

static_assert(std::is_assignable_v<
    const std::pair<AssignableFrom<ConstCopyAssign>, AssignableFrom<ConstCopyAssign>>&,
    const std::pair<ConstCopyAssign, ConstCopyAssign>&>);

static_assert(!std::is_assignable_v<
    const std::pair<AssignableFrom<CopyAssign>, AssignableFrom<CopyAssign>>&,
    const std::pair<CopyAssign, CopyAssign>&>);
// clang-format on

constexpr bool test() {
  // reference types
  {
    int i1  = 1;
    int i2  = 2;
    long j1 = 3;
    long j2 = 4;
    const std::pair<int&, int&> p1{i1, i2};
    const std::pair<long&, long&> p2{j1, j2};
    p2 = p1;
    assert(p2.first == 1);
    assert(p2.second == 2);
  }

  // user defined const copy assignment
  {
    const std::pair<ConstCopyAssign, ConstCopyAssign> p1{1, 2};
    const std::pair<AssignableFrom<ConstCopyAssign>, AssignableFrom<ConstCopyAssign>> p2{3, 4};
    p2 = p1;
    assert(p2.first.v.val == 1);
    assert(p2.second.v.val == 2);
  }

  // The correct assignment operator of the underlying type is used
  {
    std::pair<TracedAssignment, TracedAssignment> t1{};
    const std::pair<AssignableFrom<TracedAssignment>, AssignableFrom<TracedAssignment>> t2{};
    t2 = t1;
    assert(t2.first.v.constCopyAssign == 1);
    assert(t2.second.v.constCopyAssign == 1);
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
