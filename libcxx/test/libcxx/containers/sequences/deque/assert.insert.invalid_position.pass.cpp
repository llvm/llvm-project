//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <deque>

// Make sure insert and emplace catch a position that isn't in [begin(), end()], such as an iterator
// into another deque or an iterator to an element that was removed.

#include <deque>
#include <vector>

#include "check_assertion.h"
#include "min_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Deque>
void test_invalid_position(Deque& d, typename Deque::const_iterator pos) {
  [[maybe_unused]] const char* message = "deque::insert/emplace called with an iterator outside [begin(), end()]";
  int a[]                              = {1, 2, 3};
  const int x                          = 4;
  TEST_LIBCPP_ASSERT_FAILURE(d.insert(pos, x), message);
  TEST_LIBCPP_ASSERT_FAILURE(d.insert(pos, 4), message);
  TEST_LIBCPP_ASSERT_FAILURE(d.insert(pos, 2, x), message);
  TEST_LIBCPP_ASSERT_FAILURE(d.insert(pos, a, a + 3), message);
  TEST_LIBCPP_ASSERT_FAILURE(d.insert(pos, forward_iterator<int*>(a), forward_iterator<int*>(a + 3)), message);
  TEST_LIBCPP_ASSERT_FAILURE(d.insert(pos, cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 3)), message);
  TEST_LIBCPP_ASSERT_FAILURE((d.insert(pos, {1, 2, 3})), message);
  TEST_LIBCPP_ASSERT_FAILURE(d.emplace(pos, 4), message);
#if TEST_STD_VER >= 23
  TEST_LIBCPP_ASSERT_FAILURE((d.insert_range(pos, std::vector<int>{1, 2, 3})), message);
#endif
}

template <class Deque>
void test() {
  // An iterator into another deque.
  {
    Deque d     = {1, 2, 3};
    Deque other = {4, 5, 6};
    test_invalid_position(d, other.begin());
    test_invalid_position(d, other.end());
  }

  // An iterator into another deque, inserting into an empty one.
  {
    Deque d;
    Deque other = {4, 5, 6};
    test_invalid_position(d, other.begin());
  }

  // An iterator to an element removed by pop_front().
  {
    Deque d                              = {1, 2, 3};
    typename Deque::const_iterator stale = d.begin();
    d.pop_front();
    test_invalid_position(d, stale);
  }

  // An iterator past end() after pop_back().
  {
    Deque d                              = {1, 2, 3};
    typename Deque::const_iterator stale = d.end();
    d.pop_back();
    test_invalid_position(d, stale);
  }
}

int main(int, char**) {
  test<std::deque<int> >();
  test<std::deque<int, min_allocator<int> > >();

  return 0;
}
