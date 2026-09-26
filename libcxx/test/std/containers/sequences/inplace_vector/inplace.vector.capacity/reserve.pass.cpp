//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// static constexpr void reserve(size_type n);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  using C = std::inplace_vector<int, 4>;
  C c{1, 2};
  auto* data = c.data();
  c.reserve(0);
  c.reserve(4);
  C::reserve(4);
  assert(c.size() == 2);
  assert(c.capacity() == 4);
  assert(c.data() == data);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  assert_throws_bad_alloc([] { std::inplace_vector<int, 4>::reserve(5); });
  std::inplace_vector<int, 4> c{1, 2};
  auto* data = c.data();
  assert_throws_bad_alloc([&] { c.reserve(5); });
  assert(c.size() == 2);
  assert(c.data() == data);
#endif

  return 0;
}
