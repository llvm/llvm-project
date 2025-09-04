//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <queue>

// template <class Alloc>
//     priority_queue(priority_queue&& q, const Alloc& a);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

template <class C>
TEST_CONSTEXPR_CXX26 C make(int n) {
  C c;
  for (int i = 0; i < n; ++i)
    c.push_back(MoveOnly(i));
  return c;
}

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
  TEST_CONSTEXPR_CXX26 Test(const value_compare& compare, container_type&& container, const test_allocator<int>& a)
      : base(compare, std::move(container), a) {}
  TEST_CONSTEXPR_CXX26 Test(Test&& q, const test_allocator<int>& a) : base(std::move(q), a) {}
  TEST_CONSTEXPR_CXX26 test_allocator<int> get_allocator() { return c.get_allocator(); }

  using base::c;
};

TEST_CONSTEXPR_CXX26 bool test() {
  Test<MoveOnly> qo(
      std::less<MoveOnly>(), make<std::vector<MoveOnly, test_allocator<MoveOnly> > >(5), test_allocator<MoveOnly>(2));
  Test<MoveOnly> q(std::move(qo), test_allocator<MoveOnly>(6));
  assert(q.size() == 5);
  assert(q.c.get_allocator() == test_allocator<MoveOnly>(6));
  assert(q.top() == MoveOnly(4));

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
