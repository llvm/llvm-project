//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none

// template <class _Iterator, class _Container, class _ContainerMaxElements>
// struct __capacity_aware_iterator;

// Check assert failure if advancing, rewinding or indexing iterator past its maximum range size
// Or if we're keeping track of the current position, if we're advancing, rewinding, indexing out of bounds.

#include <__iterator/capacity_aware_iterator.h>
#include <iterator>

#include "check_assertion.h"

// Specialization where we can fit a running count inside unused alignment bits.
template <typename Iter>
void test_bounded() {
  int arr[] = {1, 2};
  int* p    = arr;

  constexpr long sz = std::size(arr);

  using CapIter = std::__capacity_aware_iterator<Iter, decltype(p), sz>;

  CapIter it = std::__make_capacity_aware_iterator<Iter, decltype(p), sz>(Iter(arr));

  TEST_LIBCPP_ASSERT_FAILURE(
      it--, "__capacity_aware_iterator::operator--: Attempt to rewind an iterator past the start");

  TEST_LIBCPP_ASSERT_FAILURE(
      it -= 1, "__capacity_aware_iterator::operator-=: Attempt to rewind an iterator past the start");

  TEST_LIBCPP_ASSERT_FAILURE(
      it += -1, "__capacity_aware_iterator::operator+=: Attempt to rewind an iterator past the start");

  TEST_LIBCPP_ASSERT_FAILURE(
      it += (sz + 1), "__capacity_aware_iterator::operator+=: Attempt to advance an iterator past the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      it += -(sz + 1), "__capacity_aware_iterator::operator+=: Attempt to rewind an iterator past the start");

  TEST_LIBCPP_ASSERT_FAILURE(
      it -= (sz + 1), "__capacity_aware_iterator::operator-=: Attempt to rewind an iterator past the start");

  TEST_LIBCPP_ASSERT_FAILURE(
      it -= -(sz + 1), "__capacity_aware_iterator::operator-=: Attempt to advance an iterator past the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      it[sz], "__capacity_aware_iterator::operator[]: Attempt to index an iterator at or past the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      it[-sz], "__capacity_aware_iterator::operator[]: Attempt to index an iterator past the start");

  ++it;
  ++it;

  TEST_LIBCPP_ASSERT_FAILURE(
      *it, "__capacity_aware_iterator::operator*: Attempt to dereference an iterator at the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      it.operator->(), "__capacity_aware_iterator::operator->: Attempt to dereference an iterator at the end");

  TEST_LIBCPP_ASSERT_FAILURE(
      ++it, "__capacity_aware_iterator::operator++: Attempt to advance an iterator past the end");
}

template <typename Iter>
void test() {
  int arr[] = {1, 2, 3, 4};

  constexpr long sz = std::size(arr);

  using CapIter = std::__capacity_aware_iterator<Iter, decltype(arr), sz>;

  CapIter it = std::__make_capacity_aware_iterator<Iter, decltype(arr), sz>(Iter(arr));

  TEST_LIBCPP_ASSERT_FAILURE(
      it += (sz + 1),
      "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");

  TEST_LIBCPP_ASSERT_FAILURE(
      it += -(sz + 1),
      "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");

  TEST_LIBCPP_ASSERT_FAILURE(
      it -= (sz + 1),
      "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");

  TEST_LIBCPP_ASSERT_FAILURE(
      it -= -(sz + 1),
      "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");

  TEST_LIBCPP_ASSERT_FAILURE(
      it[sz],
      "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");

  TEST_LIBCPP_ASSERT_FAILURE(
      it[-sz],
      "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");
}

int main(int, char**) {
  test_bounded<int*>();
  test<int*>();

  return 0;
}
