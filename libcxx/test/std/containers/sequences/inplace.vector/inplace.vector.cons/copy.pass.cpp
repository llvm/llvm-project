//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector(const inplace_vector& v);

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"

template <class C>
constexpr void test(const C& x) {
  typename C::size_type s = x.size();
  C c(x);
  assert(c.size() == s);
  assert(c == x);
}

constexpr bool tests() {
  {
    int a[]             = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 1, 0};
    constexpr auto a_sz = sizeof(a) / sizeof(a[0]);
    int* an             = a + a_sz;
    test(std::inplace_vector<int, a_sz>(a, an));
    test(std::inplace_vector<int, a_sz + 1>(a, an));
    test(std::inplace_vector<int, a_sz * 2>(a, an));
  }
  {
    // Test copy ctor with empty source
    std::inplace_vector<int, 10 > v;
    std::inplace_vector<int, 10> v2 = v;
    assert(v2 == v);
    assert(v2.empty());
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
