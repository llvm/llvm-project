//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class InputIterator>
//   constexpr void assign(InputIterator first, InputIterator last);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
constexpr void test_iterators() {
  int a[] = {1, 2, 3, 4};
  std::inplace_vector<int, 8> c{9, 8, 7};
  c.assign(Iter(a), Iter(a + 4));
  assert_inplace_vector_equal(c, a);

  int b[] = {5};
  c.assign(Iter(b), Iter(b + 1));
  assert_inplace_vector_equal(c, b);

  c.assign(Iter(b), Iter(b));
  assert(c.empty());
}

constexpr bool test() {
  test_iterators<cpp17_input_iterator<int*> >();
  test_iterators<forward_iterator<int*> >();
  test_iterators<int*>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  int a[] = {1, 2, 3, 4, 5};
  std::inplace_vector<int, 4> c{9, 8};
  assert_throws_bad_alloc([&] { c.assign(a, a + 5); });
  assert_inplace_vector_equal(c, {9, 8});
#endif

  return 0;
}
