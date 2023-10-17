//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Call back() on empty container.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=unchecked
// XFAIL: availability-verbose_abort-missing

#include <vector>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector<T, min_allocator<T> > C;
    C c(1);
    assert(c.back() == 0);
    c.clear();
    TEST_LIBCPP_ASSERT_FAILURE(c.back(), "back() called on an empty vector");
  }
  {
    typedef int T;
    typedef std::vector<T> C;
    C c(1);
    assert(c.back() == 0);
    c.clear();
    TEST_LIBCPP_ASSERT_FAILURE(c.back(), "back() called on an empty vector");
  }

  return 0;
}
