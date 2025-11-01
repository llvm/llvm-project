//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stack>

// template <class T, class Container>
//   bool operator< (const stack<T, Container>& x,const stack<T, Container>& y);
//
// template <class T, class Container>
//   bool operator> (const stack<T, Container>& x,const stack<T, Container>& y);
//
// template <class T, class Container>
//   bool operator>=(const stack<T, Container>& x,const stack<T, Container>& y);
//
// template <class T, class Container>
//   bool operator<=(const stack<T, Container>& x,const stack<T, Container>& y);

#include <stack>
#include <cassert>

#include "test_macros.h"

template <class C>
TEST_CONSTEXPR_CXX26 C make(int n) {
  C c;
  for (int i = 0; i < n; ++i)
    c.push(i);
  return c;
}

TEST_CONSTEXPR_CXX26 bool test() {
  std::stack<int> q1 = make<std::stack<int> >(5);
  std::stack<int> q2 = make<std::stack<int> >(10);
  assert(q1 < q2);
  assert(q2 > q1);
  assert(q1 <= q2);
  assert(q2 >= q1);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
