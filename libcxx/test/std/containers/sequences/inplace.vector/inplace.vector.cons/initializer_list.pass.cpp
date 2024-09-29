//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector(initializer_list<value_type> il);

#include <inplace_vector>
#include <cassert>
#include "test_macros.h"

constexpr bool tests() {
  {
    std::inplace_vector<int, 10> d = {3, 4, 5, 6};
    assert(d.size() == 4);
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
  }
  {
    std::inplace_vector<int, 4> d = {3, 4, 5, 6};
    assert(d.size() == 4);
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
  }
  {
    std::inplace_vector<int, 10> d = std::initializer_list<int>();
    assert(d.size() == 0);
  }
  {
    std::inplace_vector<int, 0> d = std::initializer_list<int>();
    assert(d.size() == 0);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
