//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <forward_list>

// void splice_after(const_iterator position, forward_list& x);
// void splice_after(const_iterator position, forward_list& x, const_iterator first, const_iterator last);
//
// Passing x == *this, or a position inside (first, last), is diagnosed.

#include <cassert>
#include <forward_list>
#include <iterator>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  {
    std::forward_list<int> a{1, 2, 3};
    TEST_LIBCPP_ASSERT_FAILURE(a.splice_after(a.before_begin(), a),
                               "forward_list::splice_after(iterator, forward_list) called with this == &forward_list");
    TEST_LIBCPP_ASSERT_FAILURE(a.splice_after(a.before_begin(), std::move(a)),
                               "forward_list::splice_after(iterator, forward_list) called with this == &forward_list");
  }

  {
    std::forward_list<int> a{1, 2, 3, 4};
    TEST_LIBCPP_ASSERT_FAILURE(
        a.splice_after(std::next(a.begin()), a, a.begin(), std::next(a.begin(), 3)),
        "forward_list::splice_after(iterator, forward_list, iterator, iterator) called with the position inside the "
        "range");
    TEST_LIBCPP_ASSERT_FAILURE(
        a.splice_after(std::next(a.begin(), 2), a, a.begin(), a.end()),
        "forward_list::splice_after(iterator, forward_list, iterator, iterator) called with the position inside the "
        "range");
  }

  { // Valid calls do not trigger the assertion.
    std::forward_list<int> a{1, 2};
    std::forward_list<int> b{3, 4};
    a.splice_after(a.before_begin(), b);
    assert((a == std::forward_list<int>{3, 4, 1, 2}));
    assert(b.empty());
  }

  { // Position before the range.
    std::forward_list<int> a{1, 2, 3, 4};
    a.splice_after(a.before_begin(), a, a.begin(), a.end());
    assert((a == std::forward_list<int>{2, 3, 4, 1}));
  }

  { // Position is last.
    std::forward_list<int> a{1, 2, 3, 4};
    a.splice_after(std::next(a.begin(), 2), a, a.begin(), std::next(a.begin(), 2));
    assert((a == std::forward_list<int>{1, 3, 2, 4}));
  }

  return 0;
}
