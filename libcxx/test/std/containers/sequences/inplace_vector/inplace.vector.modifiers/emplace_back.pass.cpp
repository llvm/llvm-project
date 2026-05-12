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
//   constexpr reference emplace_back(Args&&... args);

#include <cassert>
#include <inplace_vector>
#include <tuple>

#include "../common.h"
#include "test_macros.h"

struct A {
  int i;
  int j;

  constexpr A(int ii, int jj) : i(ii), j(jj) {}
};

constexpr bool test() {
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<A, 3> c;
    ASSERT_SAME_TYPE(A&, decltype(c.emplace_back(1, 2)));
    A& r = c.emplace_back(1, 2);
    assert(&r == &c.back());
    assert(c.back().i == 1);
    assert(c.back().j == 2);

    c.emplace_back(3, 4);
    assert(c.size() == 2);
    assert(c[1].i == 3);
    assert(c[1].j == 4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 2> c{1, 2};
  assert_throws_bad_alloc([&] { c.emplace_back(3); });
  assert_inplace_vector_equal(c, {1, 2});
#endif

  return 0;
}
