//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class T, class Allocator>
//     bool operator< (const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y); // constexpr since C++26
//
// template <class T, class Allocator>
//     bool operator> (const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y); // constexpr since C++26
//
// template <class T, class Allocator>
//     bool operator>=(const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y); // constexpr since C++26
//
// template <class T, class Allocator>
//     bool operator<=(const forward_list<T, Allocator>& x,
//                     const forward_list<T, Allocator>& y); // constexpr since C++26

#include <forward_list>
#include <iterator>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class C>
TEST_CONSTEXPR_CXX26 void test(int N, int M) {
  C c1;
  for (int i = 0; i < N; ++i)
    c1.push_front(i);
  C c2;
  for (int i = 0; i < M; ++i)
    c2.push_front(i);
  if (N < M)
    assert(c1 < c2);
  if (N <= M)
    assert(c1 <= c2);
  if (N >= M)
    assert(c1 >= c2);
  if (N > M)
    assert(c1 > c2);
}

TEST_CONSTEXPR_CXX26 bool test() {
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      test<std::forward_list<int> >(i, j);
#if TEST_STD_VER >= 11
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      test<std::forward_list<int, min_allocator<int>> >(i, j);
#endif

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
