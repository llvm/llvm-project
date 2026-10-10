//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr void pop_back();

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 4> c{1, 2, 3};
  c.pop_back();
  assert_inplace_vector_equal(c, {1, 2});
  c.pop_back();
  assert_inplace_vector_equal(c, {1});
  c.pop_back();
  assert(c.empty());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
