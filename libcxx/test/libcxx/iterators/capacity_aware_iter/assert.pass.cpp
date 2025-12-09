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

// Check assert failure if advancing, rewinding or indexing iterator past _ContainerMaxElements

#include <__iterator/capacity_aware_iterator.h>
#include <iterator>

#include "check_assertion.h"
#include "test_iterators.h"
#include "test_macros.h"

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
  test<cpp20_random_access_iterator<int*>>();

  return 0;
}
