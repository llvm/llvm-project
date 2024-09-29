//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// void assign(size_type n, const_reference v);

#include <inplace_vector>
#include <algorithm>
#include <cassert>

#include "test_macros.h"

constexpr bool is6(int x) { return x == 6; }

template <typename Vec>
constexpr void test(Vec& v) {
  v.assign(5, 6);
  assert(v.size() == 5);
  assert(std::all_of(v.begin(), v.end(), is6));
}

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 10>;
    V d1;
    V d2{1, 2, 3, 4, 5, 6, 7, 8};
    V d3{1, 2, 3};
    test(d1);
    test(d2);
    test(d3);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
