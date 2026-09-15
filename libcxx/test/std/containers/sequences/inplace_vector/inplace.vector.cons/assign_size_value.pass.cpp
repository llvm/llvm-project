//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr void assign(size_type n, const T& u);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 8> c{1, 2, 3};
  c.assign(4, 7);
  assert_inplace_vector_equal(c, {7, 7, 7, 7});
  c.assign(1, 8);
  assert_inplace_vector_equal(c, {8});
  c.assign(0, 9);
  assert(c.empty());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 3> c{1, 2};
  assert_throws_bad_alloc([&] { c.assign(4, 7); });
  assert_inplace_vector_equal(c, {1, 2});
#endif

  return 0;
}
