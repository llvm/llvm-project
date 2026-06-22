//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr reference push_back(const T& x);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 3> c;
  int x = 1;
  ASSERT_SAME_TYPE(int&, decltype(c.push_back(x)));
  int& r1 = c.push_back(x);
  assert(&r1 == &c.back());
  assert(c.back() == 1);

  x = 2;

  int& r2 = c.push_back(x);
  assert(&r2 == &c.back());
  assert_inplace_vector_equal(c, {1, 2});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 2> c{1, 2};
  int x = 3;
  assert_throws_bad_alloc([&] { c.push_back(x); });
  assert_inplace_vector_equal(c, {1, 2});
#endif

  return 0;
}
