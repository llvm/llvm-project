//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class Alloc>
//     explicit priority_queue(const Alloc& a);

#include <queue>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

template <class T>
struct Test : public std::priority_queue<T, std::vector<T, test_allocator<T> > > {
  typedef std::priority_queue<T, std::vector<T, test_allocator<T> > > base;
  typedef typename base::container_type container_type;
  typedef typename base::value_compare value_compare;

  TEST_CONSTEXPR_CXX26 explicit Test(const test_allocator<int>& a) : base(a) {}
  TEST_CONSTEXPR_CXX26 Test(const value_compare& comp, const test_allocator<int>& a) : base(comp, c, a) {}
  TEST_CONSTEXPR_CXX26 Test(const value_compare& comp, const container_type& container, const test_allocator<int>& a)
      : base(comp, container, a) {}
#if TEST_STD_VER >= 11
  TEST_CONSTEXPR_CXX26 Test(const value_compare& comp, container_type&& container, const test_allocator<int>& a)
      : base(comp, std::move(container), a) {}
  TEST_CONSTEXPR_CXX26 Test(Test&& q, const test_allocator<int>& a) : base(std::move(q), a) {}
#endif
  TEST_CONSTEXPR_CXX26 test_allocator<int> get_allocator() { return c.get_allocator(); }

  using base::c;
};

TEST_CONSTEXPR_CXX26 bool test() {
  Test<int> q((test_allocator<int>(3)));
  assert(q.c.get_allocator() == test_allocator<int>(3));
  assert(q.c.size() == 0);

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
