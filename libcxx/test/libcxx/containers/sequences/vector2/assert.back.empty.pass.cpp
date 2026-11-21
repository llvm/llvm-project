//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// Call back() on empty container.

// REQUIRES: can-test-hardening-assertions-fast

#include <vector>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector2<T, min_allocator<T> > C;
    C c(1);
    assert(c.back() == 0);
    c.clear();
    TEST_LIBCPP_ASSERT_FAILURE(c.back(), "back() called on an empty vector2");
  }
  {
    typedef int T;
    typedef std::vector2<T> C;
    C c(1);
    assert(c.back() == 0);
    c.clear();
    TEST_LIBCPP_ASSERT_FAILURE(c.back(), "back() called on an empty vector2");
  }

  return 0;
}
