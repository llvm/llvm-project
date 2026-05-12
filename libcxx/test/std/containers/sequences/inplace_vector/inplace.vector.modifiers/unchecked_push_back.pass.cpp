//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class... Args>
//   constexpr reference unchecked_emplace_back(Args&&... args);
// constexpr reference unchecked_push_back(const T& x);
// constexpr reference unchecked_push_back(T&& x);

#include <cassert>
#include <inplace_vector>
#include <utility>

#include "../common.h"
#include "MoveOnly.h"
#include "test_macros.h"

struct A {
  int i;
  int j;

  constexpr A(int ii, int jj) : i(ii), j(jj) {}
};

constexpr bool test() {
  {
    std::inplace_vector<int, 3> c;
    int value = 1;
    ASSERT_SAME_TYPE(int&, decltype(c.unchecked_push_back(value)));
    ASSERT_SAME_TYPE(int&, decltype(c.unchecked_push_back(1)));
    int& r1 = c.unchecked_push_back(value);
    assert(&r1 == &c.back());
    int& r2 = c.unchecked_push_back(2);
    assert(&r2 == &c.back());
    assert_inplace_vector_equal(c, {1, 2});
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<A, 2> c;
    ASSERT_SAME_TYPE(A&, decltype(c.unchecked_emplace_back(1, 2)));
    A& r = c.unchecked_emplace_back(1, 2);
    assert(&r == &c.back());
    assert(c[0].i == 1);
    assert(c[0].j == 2);
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<MoveOnly, 1> c;
    MoveOnly& r = c.unchecked_push_back(MoveOnly(1));
    assert(&r == &c.back());
    assert(c[0].get() == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
