//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector(size_type n, const T& value);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  {
    std::inplace_vector<int, 5> c(0, 7);
    assert(c.empty());
  }
  {
    std::inplace_vector<int, 5> c(3, 7);
    assert_inplace_vector_equal(c, {7, 7, 7});
  }
  {
    std::inplace_vector<int, 5> c(5, 7);
    assert_inplace_vector_equal(c, {7, 7, 7, 7, 7});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

#ifndef TEST_HAS_NO_EXCEPTIONS
  assert_throws_bad_alloc([] { std::inplace_vector<int, 5> c(6, 7); });
#endif

  return 0;
}
