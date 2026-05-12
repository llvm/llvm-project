//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers, std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

// <inplace_vector>

// Check invalid iterator operations.

#include <inplace_vector>

#include "check_assertion.h"

int main(int, char**) {
#ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_INPLACE_VECTOR
  { // operator++
    std::inplace_vector<int, 4> c{1};
    auto i = c.end();

    TEST_LIBCPP_ASSERT_FAILURE(++i, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i++, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
  }

  { // operator--
    std::inplace_vector<int, 4> c{1};
    auto i = c.begin();

    TEST_LIBCPP_ASSERT_FAILURE(--i, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i--, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
  }

  { // operator*
    std::inplace_vector<int, 4> c;
    auto i = c.begin();

    TEST_LIBCPP_ASSERT_FAILURE(*i, "__bounded_iter::operator*: Attempt to dereference an iterator at the end");
  }

  { // operator[]
    std::inplace_vector<int, 4> c{1};
    auto i = c.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i[1], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i[-1], "__bounded_iter::operator[]: Attempt to index an iterator past the start");
  }

  { // operator->
    std::inplace_vector<int, 4> c{1};
    auto i = c.end();

    TEST_LIBCPP_ASSERT_FAILURE(
        i.operator->(), "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
  }

  { // operator+=
    std::inplace_vector<int, 4> c{1};
    auto i = c.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i += 2, "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i += -1, "__bounded_iter::operator+=: Attempt to rewind an iterator past the start");
  }

  { // operator-=
    std::inplace_vector<int, 4> c{1};
    auto i = c.begin();

    TEST_LIBCPP_ASSERT_FAILURE(i -= 1, "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i -= -2, "__bounded_iter::operator-=: Attempt to advance an iterator past the end");
  }

  {
    std::inplace_vector<int, 0> c;

    auto i = c.begin();
    TEST_LIBCPP_ASSERT_FAILURE(++i, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i++, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(--i, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i--, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i[0], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i[0], "__bounded_iter::operator[]: Attempt to index an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(
        i.operator->(), "__bounded_iter::operator->: Attempt to dereference an iterator at the end");
    TEST_LIBCPP_ASSERT_FAILURE(i += 1, "__bounded_iter::operator+=: Attempt to advance an iterator past the end");
    TEST_LIBCPP_ASSERT_FAILURE(i += -1, "__bounded_iter::operator+=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i -= 1, "__bounded_iter::operator-=: Attempt to rewind an iterator past the start");
    TEST_LIBCPP_ASSERT_FAILURE(i -= -1, "__bounded_iter::operator-=: Attempt to advance an iterator past the end");
  }
#else
  std::inplace_vector<int, 4> c{1};
  auto i = c.begin();

  TEST_LIBCPP_ASSERT_FAILURE(
      i += 5, "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");
  TEST_LIBCPP_ASSERT_FAILURE(
      i += -5,
      "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");
  TEST_LIBCPP_ASSERT_FAILURE(
      i -= 5, "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");
  TEST_LIBCPP_ASSERT_FAILURE(
      i -= -5,
      "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");
  TEST_LIBCPP_ASSERT_FAILURE(
      i[4], "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");
  TEST_LIBCPP_ASSERT_FAILURE(
      i[-5], "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");
#endif

  return 0;
}
