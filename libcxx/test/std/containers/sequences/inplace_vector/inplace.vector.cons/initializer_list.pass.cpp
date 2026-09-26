//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector(initializer_list<T> il);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 8> c{1, 2, 3};
  assert_inplace_vector_equal(c, {1, 2, 3});

  std::inplace_vector<int, 0> empty{};
  assert(empty.empty());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  assert_throws_bad_alloc([] { std::inplace_vector<int, 2> c{1, 2, 3}; });
#endif

  return 0;
}
