//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr iterator erase(const_iterator first, const_iterator last);

#include <cassert>
#include <concepts>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  {
    std::inplace_vector<int, 8> c{1, 2, 3, 4, 5};
    std::same_as<std::inplace_vector<int, 8>::iterator> decltype(auto) i = c.erase(c.begin(), c.begin() + 2);
    assert(i == c.begin());
    assert_inplace_vector_equal(c, {3, 4, 5});
  }
  {
    std::inplace_vector<int, 8> c{1, 2, 3, 4, 5};
    auto i = c.erase(c.begin() + 1, c.begin() + 4);
    assert(i == c.begin() + 1);
    assert_inplace_vector_equal(c, {1, 5});
  }
  {
    std::inplace_vector<int, 8> c{1, 2, 3, 4, 5};
    auto i = c.erase(c.end(), c.end());
    assert(i == c.end());
    assert_inplace_vector_equal(c, {1, 2, 3, 4, 5});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
