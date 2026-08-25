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
//   constexpr inplace_vector(InputIterator first, InputIterator last);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
constexpr void test_iterators() {
  int a[] = {1, 2, 3, 4};
  std::inplace_vector<int, 8> c(Iter(a), Iter(a + 4));
  assert_inplace_vector_equal(c, a);
}

constexpr bool test() {
  test_iterators<cpp17_input_iterator<int*> >();
  test_iterators<forward_iterator<int*> >();
  test_iterators<int*>();

  {
    int a[] = {1, 2, 3, 4};
    std::inplace_vector<int, 4> c(a, a + 4);
    assert_inplace_vector_equal(c, a);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  int a[] = {1, 2, 3, 4, 5};
  assert_throws_bad_alloc([&] { std::inplace_vector<int, 4> c(a, a + 5); });
#endif

  return 0;
}
