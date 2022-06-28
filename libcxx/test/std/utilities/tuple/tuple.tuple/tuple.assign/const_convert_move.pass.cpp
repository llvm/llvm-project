//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... UTypes>
// constexpr const tuple& operator=(tuple<UTypes...>&& u) const;
//
// Constraints:
// - sizeof...(Types) equals sizeof...(UTypes) and
// - (is_assignable_v<const Types&, UTypes> && ...) is true.

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <tuple>
#include <type_traits>

#include "test_macros.h"
#include "types.h"

// test constraints

// sizeof...(Types) equals sizeof...(UTypes)
static_assert(std::is_assignable_v<const std::tuple<int&>&, std::tuple<long&>&&>);
static_assert(!std::is_assignable_v<const std::tuple<int&, int&>&, std::tuple<long&>&&>);
static_assert(!std::is_assignable_v<const std::tuple<int&>&, std::tuple<long&, long&>&&>);

// (is_assignable_v<const Types&, UTypes&&> && ...) is true
static_assert(std::is_assignable_v<const std::tuple<AssignableFrom<ConstMoveAssign>>&, std::tuple<ConstMoveAssign>&&>);

static_assert(std::is_assignable_v<const std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign>&,
                                   std::tuple<ConstMoveAssign, ConstMoveAssign>&&>);

static_assert(!std::is_assignable_v<const std::tuple<AssignableFrom<ConstMoveAssign>, AssignableFrom<MoveAssign>>&,
                                    std::tuple<ConstMoveAssign, MoveAssign>&&>);

constexpr bool test() {
  // reference types
  {
    int i1 = 1;
    int i2 = 2;
    long j1 = 3;
    long j2 = 4;
    std::tuple<int&, int&> t1{i1, i2};
    const std::tuple<long&, long&> t2{j1, j2};
    t2 = std::move(t1);
    assert(std::get<0>(t2) == 1);
    assert(std::get<1>(t2) == 2);
  }

  // user defined const copy assignment
  {
    std::tuple<ConstMoveAssign> t1{1};
    const std::tuple<AssignableFrom<ConstMoveAssign>> t2{2};
    t2 = std::move(t1);
    assert(std::get<0>(t2).v.val == 1);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    std::tuple<TracedAssignment> t1{};
    const std::tuple<AssignableFrom<TracedAssignment>> t2{};
    t2 = std::move(t1);
    assert(std::get<0>(t2).v.constMoveAssign == 1);
  }

  return true;
}

int main(int, char**) {
  test();

// gcc cannot have mutable member in constant expression
#if !defined(TEST_COMPILER_GCC)
  static_assert(test());
#endif
  return 0;
}
