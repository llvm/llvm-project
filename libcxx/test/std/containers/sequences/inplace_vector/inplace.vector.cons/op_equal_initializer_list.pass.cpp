//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector& operator=(initializer_list<T>);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  using C = std::inplace_vector<int, 8>;
  C c{1, 2, 3};
  ASSERT_SAME_TYPE(C&, decltype(c = {4, 5}));
  C& result = (c = {4, 5});
  assert(&result == &c);
  assert_inplace_vector_equal(c, {4, 5});
  c = {};
  assert(c.empty());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 3> c{1, 2};
  assert_throws_bad_alloc([&] { c = {1, 2, 3, 4}; });
  assert_inplace_vector_equal(c, {1, 2});
#endif

  return 0;
}
