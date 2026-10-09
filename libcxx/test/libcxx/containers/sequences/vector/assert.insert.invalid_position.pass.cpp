//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Make sure insert and emplace catch a position that isn't in [begin(), end()], such as an iterator
// into another vector or an iterator invalidated by a reallocation.

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

#include <vector>

#include "check_assertion.h"
#include "min_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Vector>
void test_invalid_position(Vector& v, typename Vector::const_iterator pos) {
  [[maybe_unused]] const char* message = "vector::insert/emplace called with an iterator outside [begin(), end()]";
  int a[]                              = {1, 2, 3};
  const int x                          = 4;
  TEST_LIBCPP_ASSERT_FAILURE(v.insert(pos, x), message);
  TEST_LIBCPP_ASSERT_FAILURE(v.insert(pos, 4), message);
  TEST_LIBCPP_ASSERT_FAILURE(v.insert(pos, 2, x), message);
  TEST_LIBCPP_ASSERT_FAILURE(v.insert(pos, a, a + 3), message);
  TEST_LIBCPP_ASSERT_FAILURE(v.insert(pos, forward_iterator<int*>(a), forward_iterator<int*>(a + 3)), message);
  TEST_LIBCPP_ASSERT_FAILURE(v.insert(pos, cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 3)), message);
  TEST_LIBCPP_ASSERT_FAILURE((v.insert(pos, {1, 2, 3})), message);
  TEST_LIBCPP_ASSERT_FAILURE(v.emplace(pos, 4), message);
#if TEST_STD_VER >= 23
  TEST_LIBCPP_ASSERT_FAILURE((v.insert_range(pos, std::vector<int>{1, 2, 3})), message);
#endif
}

template <class Vector>
void test() {
  // An iterator into another vector.
  {
    Vector v     = {1, 2, 3};
    Vector other = {4, 5, 6};
    test_invalid_position(v, other.begin());
    test_invalid_position(v, other.end());
  }

  // An iterator into another vector, inserting into an empty one.
  {
    Vector v;
    Vector other = {4, 5, 6};
    test_invalid_position(v, other.begin());
  }

  // An iterator invalidated by a reallocation.
  {
    Vector v                              = {1, 2, 3};
    typename Vector::const_iterator stale = v.begin();
    v.reserve(v.capacity() + 1);
    test_invalid_position(v, stale);
  }

  // An iterator past end() that is still inside the allocation.
  {
    Vector v = {1, 2, 3};
    v.reserve(10);
    typename Vector::const_iterator stale = v.end();
    v.pop_back();
    test_invalid_position(v, stale);
  }
}

int main(int, char**) {
  test<std::vector<int> >();
  test<std::vector<int, min_allocator<int> > >();

  return 0;
}
