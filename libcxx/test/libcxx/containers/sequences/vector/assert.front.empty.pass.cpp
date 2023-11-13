//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Call front() on empty container.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
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
    assert(c.front() == 0);
    c.clear();
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "front() called on an empty vector");
  }

  {
    typedef int T;
    typedef std::vector<T> C;
    C c(1);
    assert(c.front() == 0);
    c.clear();
    TEST_LIBCPP_ASSERT_FAILURE(c.front(), "front() called on an empty vector");
  }

  return 0;
}
