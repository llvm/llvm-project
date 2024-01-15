//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: availability-verbose_abort-missing

#include <iterator>

#include "check_assertion.h"
#include "test_iterators.h"

int main(int, char**) {
  using Iter = std::common_iterator<int*, sentinel_wrapper<int*>>;
  int a[]    = {1, 2, 3};
  sentinel_wrapper<int*> s;
  Iter valid_i = a;

  {
    Iter i = s;

    TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable common_iterator");

    TEST_LIBCPP_ASSERT_FAILURE(++i, "Attempted to increment a non-dereferenceable common_iterator");
    TEST_LIBCPP_ASSERT_FAILURE(i++, "Attempted to increment a non-dereferenceable common_iterator");

    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::iter_move(i), "Attempted to iter_move a non-dereferenceable common_iterator");

    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::iter_swap(i, valid_i), "Attempted to iter_swap a non-dereferenceable common_iterator");
    TEST_LIBCPP_ASSERT_FAILURE(
        std::ranges::iter_swap(valid_i, i), "Attempted to iter_swap a non-dereferenceable common_iterator");
    std::ranges::iter_swap(valid_i, valid_i); // Ok
  }

  { // Check the `const` overload of `operator*`.
    const Iter i = s;
    TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable common_iterator");
  }

  { // Check `operator->`.
    struct Foo {
      int x = 0;
    };

    std::common_iterator<Foo*, sentinel_wrapper<Foo*>> i = sentinel_wrapper<Foo*>();
    TEST_LIBCPP_ASSERT_FAILURE(i->x, "Attempted to dereference a non-dereferenceable common_iterator");
  }

  return 0;
}
