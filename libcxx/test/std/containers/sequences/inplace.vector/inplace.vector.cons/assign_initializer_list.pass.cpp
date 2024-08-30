//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// void assign(initializer_list<value_type> il);

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"

template <typename Vec>
constexpr void test(Vec& v) {
  v.assign({3, 4, 5, 6});
  assert(v.size() == 4);
  assert(v[0] == 3);
  assert(v[1] == 4);
  assert(v[2] == 5);
  assert(v[3] == 6);
  v.assign({});
  assert(v.size() == 0);
}

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 4>;
    V d1;
    V d2{-1, -2};
    V d3{-1, -2, -3, -4};
    test(d1);
    test(d2);
    test(d3);
  }
  {
    using V = std::inplace_vector<int, 10>;
    V d1;
    V d2{-1, -2};
    V d3{-1, -2, -3, -4};
    V d4{-1, -2, -3, -4, -6, -7};
    test(d1);
    test(d2);
    test(d3);
    test(d4);
  }
  {
    using V = std::inplace_vector<int, 100>;
    V d1;
    V d2{-1, -2};
    V d3{-1, -2, -3, -4};
    V d4{-1, -2, -3, -4, -6, -7};
    test(d1);
    test(d2);
    test(d3);
    test(d4);
  }
  {
    using V = std::inplace_vector<int, 0>;
    V d1;
    d1.assign({});
    assert(d1.size() == 0);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
