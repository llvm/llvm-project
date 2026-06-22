//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr void resize(size_type sz, const T& c);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 5> c{1, 2, 3};
  c.resize(5, 9);
  assert_inplace_vector_equal(c, {1, 2, 3, 9, 9});
  c.resize(2, 7);
  assert_inplace_vector_equal(c, {1, 2});
  c.resize(5, 8);
  assert_inplace_vector_equal(c, {1, 2, 8, 8, 8});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 5> c{1, 2, 3};
  assert_throws_bad_alloc([&] { c.resize(6, 4); });
  assert_inplace_vector_equal(c, {1, 2, 3});
#endif

  return 0;
}
