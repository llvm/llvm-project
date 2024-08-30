//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// An inplace_vector is a contiguous container

#include <inplace_vector>
#include <memory>
#include <cassert>

#include "test_macros.h"

template <class C>
constexpr void test_contiguous(const C& c) {
  for (std::size_t i = 0; i <= c.size(); ++i) {
    if (i != c.size()) {
      assert(*(c.begin() + i) == *(std::addressof(*c.begin()) + i));
      assert(std::addressof(c.begin()[i]) == std::addressof(c.front()) + i);
    }
    assert(std::to_address(c.begin() + i) == std::to_address(c.begin()) + i);
  }
}

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 10>;
    test_contiguous(V());
    test_contiguous(V(3, 5));
  }
  {
    using V = std::inplace_vector<int, 0>;
    test_contiguous(V());
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
