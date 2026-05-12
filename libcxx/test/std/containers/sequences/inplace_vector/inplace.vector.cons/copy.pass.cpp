//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector(const inplace_vector&);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  {
    std::inplace_vector<int, 8> c;
    std::inplace_vector<int, 8> copy(c);
    assert(copy.empty());
  }
  {
    std::inplace_vector<int, 8> c{1, 2, 3};
    std::inplace_vector<int, 8> copy(c);
    assert_inplace_vector_equal(copy, {1, 2, 3});
    assert_inplace_vector_equal(c, {1, 2, 3});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
