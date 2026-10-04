//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr iterator insert(const_iterator position, initializer_list<T> il);

#include <cassert>
#include <concepts>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 8> c{1, 2, 5};
  std::same_as<std::inplace_vector<int, 8>::iterator> decltype(auto) i = c.insert(c.begin() + 2, {3, 4});
  assert(i == c.begin() + 2);
  assert_inplace_vector_equal(c, {1, 2, 3, 4, 5});

  i = c.insert(c.end(), {});
  assert(i == c.end());
  assert_inplace_vector_equal(c, {1, 2, 3, 4, 5});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 4> c{1, 2, 3};
  assert_throws_bad_alloc([&] { c.insert(c.begin(), {4, 5}); });
  assert_inplace_vector_equal(c, {1, 2, 3});
#endif

  return 0;
}
