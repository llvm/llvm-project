//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// Test hardening assertions for std::forward_list.

// REQUIRES: can-test-hardening-assertions-extensive

#include <forward_list>
#include <iterator>

#include "check_assertion.h"

int main(int, char**) {
  { // Default-constructed list.
    std::forward_list<int> c;
    const auto& const_c = c;
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "forward_list::front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(const_c.front(), "forward_list::front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(c.pop_front(), "forward_list::pop_front called on an empty list");
  }

  { // Non-empty list becomes empty.
    std::forward_list<int> c;
    const auto& const_c = c;
    c.push_front(1);

    // Check that there's no assertion on valid access.
    (void)c.front();
    (void)const_c.front();

    c.pop_front();
    TEST_LIBCPP_ASSERT_FAILURE(c.pop_front(), "forward_list::pop_front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "forward_list::front called on an empty list");
    TEST_LIBCPP_ASSERT_FAILURE(const_c.front(), "forward_list::front called on an empty list");
  }

  { // splice_after with an invalid iterator or range.
    std::forward_list<int> a{1};
    std::forward_list<int> b{4, 5};
    TEST_LIBCPP_ASSERT_FAILURE(
        a.splice_after(a.before_begin(), b, std::next(b.begin())),
        "forward_list::splice_after(iterator, forward_list, iterator) called with an iterator whose successor is not "
        "dereferenceable");
    TEST_LIBCPP_ASSERT_FAILURE(
        a.splice_after(a.before_begin(), b, b.end()),
        "forward_list::splice_after(iterator, forward_list, iterator) called with an iterator whose successor is not "
        "dereferenceable");

    std::forward_list<int> c{4, 5, 6};
    TEST_LIBCPP_ASSERT_FAILURE(
        a.splice_after(a.before_begin(), c, std::next(c.begin()), c.begin()),
        "forward_list::splice_after(iterator, forward_list, iterator, iterator) called with an invalid range");
    TEST_LIBCPP_ASSERT_FAILURE(
        a.splice_after(a.before_begin(), c, c.end(), c.begin()),
        "forward_list::splice_after(iterator, forward_list, iterator, iterator) called with an invalid range");
  }

  return 0;
}
