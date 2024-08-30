//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector& operator=(const inplace_vector& c);

#include <inplace_vector>
#include <cassert>
#include "test_macros.h"

constexpr bool tests() {
  {
    std::inplace_vector<int, 100> l(3, 2);
    std::inplace_vector<int, 100> l2(l);
    l2 = l;
    assert(l2 == l);
  }
  {
    std::inplace_vector<int, 10> l(3, 2);
    std::inplace_vector<int, 10> l2;
    l2 = l;
    assert(l2 == l);
  }
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
