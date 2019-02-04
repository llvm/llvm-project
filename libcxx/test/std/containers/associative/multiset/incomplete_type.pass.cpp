//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// Check that std::multiset and its iterators can be instantiated with an incomplete
// type.

#include <set>

struct A {
    typedef std::multiset<A> Set;
    int data;
    Set m;
    Set::iterator it;
    Set::const_iterator cit;
};

inline bool operator==(A const& L, A const& R) { return &L == &R; }
inline bool operator<(A const& L, A const& R)  { return L.data < R.data; }
int main(int, char**) {
    A a;

  return 0;
}
