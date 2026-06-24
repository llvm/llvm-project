//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map> // constexpr since C++26

// Check that std::map and its iterators can be instantiated with an incomplete
// type.

#include <map>

#include <cassert>

#include "min_allocator.h"
#include "test_macros.h"

struct A {
  typedef std::map<A, A> Map;
  int data;
  Map m;
  Map::iterator it;
  Map::const_iterator cit;
};

inline bool operator==(A const& L, A const& R) { return &L == &R; }
inline bool operator<(A const& L, A const& R) { return L.data < R.data; }
TEST_CONSTEXPR_CXX26 bool test() {
  A a;

  // Make sure that the allocator isn't rebound to and incomplete type
  std::map<int, int, std::less<int>, complete_type_allocator<std::pair<const int, int> > > m;

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
