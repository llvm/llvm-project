//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// iterator insert(const_iterator position, size_type n, const value_type& x); // constexpr since C++26

// UNSUPPORTED: sanitizer-new-delete

#include <list>
#include <cstdlib>
#include <cassert>

#include "min_allocator.h"
#include "count_new.h"
#include "test_macros.h"

template <class List>
TEST_CONSTEXPR_CXX26 void test() {
  int a1[] = {1, 2, 3};
  int a2[] = {1, 4, 4, 4, 4, 4, 2, 3};
  List l1(a1, a1 + 3);
  typename List::iterator i = l1.insert(std::next(l1.cbegin()), 5, 4);
  assert(i == std::next(l1.begin()));
  assert(l1 == List(a2, a2 + 8));
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    globalMemCounter.throw_after = 4;
    int save_count               = globalMemCounter.outstanding_new;
    try {
      i = l1.insert(i, 5, 5);
      assert(false);
    } catch (...) {
    }
    assert(globalMemCounter.checkOutstandingNewEq(save_count));
    assert(l1 == List(a2, a2 + 8));
  }
#endif

  { // Check that inserting from the same vector works as expected
    std::list<int> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);
    v.insert(v.begin(), 2, v.back());
    assert(v.size() == 6);
    int expected[] = {4, 4, 1, 2, 3, 4};
    assert(std::equal(v.begin(), v.end(), expected));
  }
}

TEST_CONSTEXPR_CXX26 bool test() {
  test<std::list<int> >();
#if TEST_STD_VER >= 11
  test<std::list<int, min_allocator<int>>>();
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
