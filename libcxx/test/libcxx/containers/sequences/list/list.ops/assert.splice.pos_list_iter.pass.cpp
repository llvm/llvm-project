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

// void splice(const_iterator position, list& x, const_iterator i);
//
// Passing an iterator that is not dereferenceable is diagnosed.

#include <cassert>
#include <list>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  {
    std::list<int> a{1, 2, 3};
    std::list<int> b{4, 5, 6};
    TEST_LIBCPP_ASSERT_FAILURE(a.splice(a.begin(), b, b.end()),
                               "list::splice(iterator, list, iterator) called with a non-dereferenceable iterator");
    TEST_LIBCPP_ASSERT_FAILURE(a.splice(a.begin(), std::move(b), b.end()),
                               "list::splice(iterator, list, iterator) called with a non-dereferenceable iterator");
  }

  {
    std::list<int> a{1, 2, 3};
    std::list<int> b;
    TEST_LIBCPP_ASSERT_FAILURE(a.splice(a.end(), b, b.begin()),
                               "list::splice(iterator, list, iterator) called with a non-dereferenceable iterator");
  }

  {
    std::list<int> a{1, 2, 3};
    TEST_LIBCPP_ASSERT_FAILURE(a.splice(a.begin(), a, a.end()),
                               "list::splice(iterator, list, iterator) called with a non-dereferenceable iterator");
  }

  { // Valid calls do not trigger the assertion.
    std::list<int> a{1, 2, 3};
    std::list<int> b{4, 5, 6};
    a.splice(a.begin(), b, b.begin());
    assert((a == std::list<int>{4, 1, 2, 3}));
    assert((b == std::list<int>{5, 6}));
    assert(a.size() == 4);
    assert(b.size() == 2);

    a.splice(a.begin(), a, a.begin());
    assert((a == std::list<int>{4, 1, 2, 3}));
    assert(a.size() == 4);
  }

  return 0;
}
