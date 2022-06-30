//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template<class U1, class U2>
// constexpr const tuple& operator=(const pair<U1, U2>& u) const;
//
// - sizeof...(Types) is 2,
// - is_assignable_v<const T1&, const U1&> is true, and
// - is_assignable_v<const T2&, const U2&> is true

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "types.h"

// test constraints

// sizeof...(Types) != 2,
static_assert(std::is_assignable_v<const std::tuple<int&, int&>&, const std::pair<int&, int&>&>);
static_assert(!std::is_assignable_v<const std::tuple<int&>&, const std::pair<int&, int&>&>);
static_assert(!std::is_assignable_v<const std::tuple<int&, int&, int&>&, const std::pair<int&, int&>&>);

static_assert(std::is_assignable_v<const std::tuple<AssignableFrom<ConstCopyAssign>, ConstCopyAssign>&,
                                   const std::pair<ConstCopyAssign, ConstCopyAssign>&>);

// is_assignable_v<const T1&, const U1&> is false
static_assert(!std::is_assignable_v<const std::tuple<AssignableFrom<CopyAssign>, ConstCopyAssign>&,
                                    const std::pair<CopyAssign, ConstCopyAssign>&>);

// is_assignable_v<const T2&, const U2&> is false
static_assert(!std::is_assignable_v<const std::tuple<AssignableFrom<ConstCopyAssign>, AssignableFrom<CopyAssign>>&,
                                    const std::tuple<ConstCopyAssign, CopyAssign>&>);

constexpr bool test() {
  // reference types
  {
    int i1 = 1;
    int i2 = 2;
    long j1 = 3;
    long j2 = 4;
    const std::pair<int&, int&> t1{i1, i2};
    const std::tuple<long&, long&> t2{j1, j2};
    t2 = t1;
    assert(std::get<0>(t2) == 1);
    assert(std::get<1>(t2) == 2);
  }

  // user defined const copy assignment
  {
    const std::pair<ConstCopyAssign, ConstCopyAssign> t1{1, 2};
    const std::tuple<AssignableFrom<ConstCopyAssign>, ConstCopyAssign> t2{3, 4};
    t2 = t1;
    assert(std::get<0>(t2).v.val == 1);
    assert(std::get<1>(t2).val == 2);
  }

  // make sure the right assignment operator of the type in the tuple is used
  {
    std::pair<TracedAssignment, TracedAssignment> t1{};
    const std::tuple<AssignableFrom<TracedAssignment>, AssignableFrom<TracedAssignment>> t2{};
    t2 = t1;
    assert(std::get<0>(t2).v.constCopyAssign == 1);
    assert(std::get<1>(t2).v.constCopyAssign == 1);
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
