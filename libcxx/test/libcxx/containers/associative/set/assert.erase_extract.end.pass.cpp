//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <set>

// Call erase(const_iterator) and extract(const_iterator) with end().

#include <set>

#include "check_assertion.h"

int main(int, char**) {
  {
    int a[] = {1, 2, 3};
    std::set<int> c(a, a + 3);
    TEST_LIBCPP_ASSERT_FAILURE(
        c.erase(c.end()), "associative container::erase(iterator) called with a non-dereferenceable iterator");
    TEST_LIBCPP_ASSERT_FAILURE(
        c.erase(c.cend()), "associative container::erase(iterator) called with a non-dereferenceable iterator");
    TEST_LIBCPP_ASSERT_FAILURE(
        c.extract(c.end()), "associative container::extract(iterator) called with a non-dereferenceable iterator");
    TEST_LIBCPP_ASSERT_FAILURE(
        c.extract(c.cend()), "associative container::extract(iterator) called with a non-dereferenceable iterator");
  }

  {
    std::set<int> c;
    TEST_LIBCPP_ASSERT_FAILURE(
        c.extract(c.end()), "associative container::extract(iterator) called with a non-dereferenceable iterator");
  }

  return 0;
}
