//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr reference push_back(T&& x);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "MoveOnly.h"
#include "test_macros.h"

constexpr bool test() {
  {
    std::inplace_vector<int, 3> c;
    ASSERT_SAME_TYPE(int&, decltype(c.push_back(1)));
    int& r = c.push_back(1);
    assert(&r == &c.back());
    assert(c.back() == 1);
    c.push_back(2);
    assert_inplace_vector_equal(c, {1, 2});
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<MoveOnly, 3> c;
    MoveOnly m(1);
    MoveOnly& r = c.push_back(std::move(m));
    assert(&r == &c.back());
    assert(c.back().get() == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 2> c{1, 2};
  assert_throws_bad_alloc([&] { c.push_back(3); });
  assert_inplace_vector_equal(c, {1, 2});
#endif

  return 0;
}
