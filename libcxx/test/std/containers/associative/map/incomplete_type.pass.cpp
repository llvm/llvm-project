//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// Check that std::map and its iterators can be instantiated with an incomplete
// type.

#include <map>

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
int main(int, char**) {
  A a;

  // Make sure that the allocator isn't rebound to and incomplete type
  std::map<int, int, std::less<int>, complete_type_allocator<std::pair<const int, int> > > m;

  return 0;
}
