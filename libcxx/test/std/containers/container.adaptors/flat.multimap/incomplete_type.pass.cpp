//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// Check that std::flat_multimap and its iterators can be instantiated with an incomplete
// type.

#include <flat_map>
#include <vector>

struct A {
  using Map = std::flat_multimap<A, A>;
  int data;
  Map m;
  Map::iterator it;
  Map::const_iterator cit;
};

// Implement the operator< required in order to instantiate flat_multimap<A, X>
bool operator<(A const& L, A const& R) { return L.data < R.data; }

int main(int, char**) {
  A a;
  return 0;
}
