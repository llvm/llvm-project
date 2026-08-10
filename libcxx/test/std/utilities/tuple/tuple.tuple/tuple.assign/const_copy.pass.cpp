//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// constexpr const tuple& operator=(const tuple&) const;
//
// Constraints: (is_copy_assignable_v<const Types> && ...) is true.

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test constraints

#include <cassert>
#include <tuple>
#include <type_traits>

#include "test_macros.h"
#include "copy_move_types.h"

static_assert(!std::is_assignable_v<const std::tuple<int>&, const std::tuple<int>&>);
static_assert(std::is_assignable_v<const std::tuple<int&>&, const std::tuple<int&>&>);
static_assert(std::is_assignable_v<const std::tuple<int&, int&>&, const std::tuple<int&, int&>&>);
static_assert(!std::is_assignable_v<const std::tuple<int&, int>&, const std::tuple<int&, int>&>);
static_assert(std::is_assignable_v<const std::tuple<ConstCopyAssign>&, const std::tuple<ConstCopyAssign>&>);
static_assert(!std::is_assignable_v<const std::tuple<CopyAssign>&, const std::tuple<CopyAssign>&>);
static_assert(!std::is_assignable_v<const std::tuple<ConstMoveAssign>&, const std::tuple<ConstMoveAssign>&>);
static_assert(!std::is_assignable_v<const std::tuple<MoveAssign>&, const std::tuple<MoveAssign>&>);

constexpr bool test() {
  // reference types
  {
    int i1 = 1;
    int i2 = 2;
    double d1 = 3.0;
    double d2 = 5.0;
    const std::tuple<int&, double&> t1{i1, d1};
    const std::tuple<int&, double&> t2{i2, d2};
    t2 = t1;
    assert(std::get<0>(t2) == 1);
    assert(std::get<1>(t2) == 3.0);
  }

  // user defined const copy assignment
  {
    const std::tuple<ConstCopyAssign> t1{1};
    const std::tuple<ConstCopyAssign> t2{2};
    t2 = t1;
    assert(std::get<0>(t2).val == 1);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    std::tuple<TracedAssignment, const TracedAssignment> t1{};
    const std::tuple<TracedAssignment, const TracedAssignment> t2{};
    t2 = t1;
    assert(std::get<0>(t2).constCopyAssign == 1);
    assert(std::get<1>(t2).constCopyAssign == 1);
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
