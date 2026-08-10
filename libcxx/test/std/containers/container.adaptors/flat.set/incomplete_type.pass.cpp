//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// Check that std::flat_set and its iterators can be instantiated with an incomplete
// type.

#include <flat_set>
#include <vector>
#include "test_macros.h"

struct A {
  using Set = std::flat_set<A>;
  int data;
  Set m;
  Set::iterator it;
  Set::const_iterator cit;
};

// Implement the operator< required in order to instantiate flat_set<A>
constexpr bool operator<(A const& L, A const& R) { return L.data < R.data; }

constexpr bool test() {
  A a;
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
