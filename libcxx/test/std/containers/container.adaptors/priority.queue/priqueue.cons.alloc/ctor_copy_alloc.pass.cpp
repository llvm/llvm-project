//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class Alloc>
//     priority_queue(const priority_queue& q, const Alloc& a);

#include <queue>
#include <cassert>

#include "test_macros.h"

template <class C>
TEST_CONSTEXPR_CXX26 C make(int n) {
  C c;
  for (int i = 0; i < n; ++i)
    c.push_back(i);
  return c;
}

#include "test_macros.h"
#include "test_allocator.h"

template <class T>
struct Test : public std::priority_queue<T, std::vector<T, test_allocator<T> > > {
  typedef std::priority_queue<T, std::vector<T, test_allocator<T> > > base;
  typedef typename base::container_type container_type;
  typedef typename base::value_compare value_compare;

  TEST_CONSTEXPR_CXX26 explicit Test(const test_allocator<int>& a) : base(a) {}
  TEST_CONSTEXPR_CXX26 Test(const value_compare& compare, const test_allocator<int>& a) : base(compare, c, a) {}
  TEST_CONSTEXPR_CXX26 Test(const value_compare& compare, const container_type& container, const test_allocator<int>& a)
      : base(compare, container, a) {}
  TEST_CONSTEXPR_CXX26 Test(const Test& q, const test_allocator<int>& a) : base(q, a) {}
  TEST_CONSTEXPR_CXX26 test_allocator<int> get_allocator() { return c.get_allocator(); }

  using base::c;
};

TEST_CONSTEXPR_CXX26 bool test() {
  Test<int> qo(std::less<int>(), make<std::vector<int, test_allocator<int> > >(5), test_allocator<int>(2));
  Test<int> q(qo, test_allocator<int>(6));
  assert(q.size() == 5);
  assert(q.c.get_allocator() == test_allocator<int>(6));
  assert(q.top() == int(4));

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
