//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class... Args>
//   constexpr iterator emplace(const_iterator position, Args&&... args);

#include <cassert>
#include <concepts>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

struct A {
  int i;
  int j;

  constexpr A(int ii, int jj) : i(ii), j(jj) {}
};

constexpr bool test() {
  {
    std::inplace_vector<int, 5> c{1, 2, 4};
    std::same_as<std::inplace_vector<int, 5>::iterator> decltype(auto) i = c.emplace(c.begin() + 2, 3);
    assert(i == c.begin() + 2);
    assert_inplace_vector_equal(c, {1, 2, 3, 4});
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    std::inplace_vector<A, 5> c;
    c.emplace_back(1, 2);
    auto i = c.emplace(c.begin(), 3, 4);
    assert(i == c.begin());
    assert(c[0].i == 3);
    assert(c[0].j == 4);
    assert(c[1].i == 1);
    assert(c[1].j == 2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  std::inplace_vector<int, 2> c{1, 2};
  assert_throws_bad_alloc([&] { c.emplace(c.begin(), 3); });
  assert_inplace_vector_equal(c, {1, 2});
#endif

  return 0;
}
