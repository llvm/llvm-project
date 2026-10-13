//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <list>

// void splice(const_iterator position, list& x, const_iterator first, const_iterator last);
//
// Passing a range that is not valid in x is diagnosed.

#include <cassert>
#include <iterator>
#include <list>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  {
    std::list<int> a{1, 2, 3};
    std::list<int> b{4, 5, 6};
    // last is before first
    TEST_LIBCPP_ASSERT_FAILURE(a.splice(a.begin(), b, std::next(b.begin()), b.begin()),
                               "list::splice(iterator, list, iterator, iterator) called with an invalid range");
    // first is end()
    TEST_LIBCPP_ASSERT_FAILURE(a.splice(a.begin(), b, b.end(), std::next(b.begin())),
                               "list::splice(iterator, list, iterator, iterator) called with an invalid range");
    TEST_LIBCPP_ASSERT_FAILURE(a.splice(a.begin(), std::move(b), std::next(b.begin()), b.begin()),
                               "list::splice(iterator, list, iterator, iterator) called with an invalid range");
  }

  { // Valid calls do not trigger the assertion.
    std::list<int> a{1, 2, 3};
    std::list<int> b{4, 5, 6};
    a.splice(a.end(), b, std::next(b.begin()), b.end());
    assert((a == std::list<int>{1, 2, 3, 5, 6}));
    assert((b == std::list<int>{4}));
    assert(a.size() == 5);
    assert(b.size() == 1);

    a.splice(a.begin(), b, b.begin(), b.begin());
    assert((a == std::list<int>{1, 2, 3, 5, 6}));
    assert((b == std::list<int>{4}));
    assert(a.size() == 5);
    assert(b.size() == 1);
  }

  return 0;
}
