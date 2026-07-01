//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set> // constexpr since C++26

// Check that std::set and its iterators can be instantiated with an incomplete
// type.

#include <set>

#include "test_macros.h"

struct A {
  typedef std::set<A> Set;
  int data;
  Set m;
  Set::iterator it;
  Set::const_iterator cit;
};

inline bool operator==(A const& L, A const& R) { return &L == &R; }
inline bool operator<(A const& L, A const& R) { return L.data < R.data; }
TEST_CONSTEXPR_CXX26 bool test() {
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
