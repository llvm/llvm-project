//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// Compare iterators from different containers with <.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode

#include <vector>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector2<T> C;
    C c1;
    C c2;
    TEST_LIBCPP_ASSERT_FAILURE(c1.begin() < c2.begin(), "Attempted to compare incomparable iterators");
  }

  {
    typedef int T;
    typedef std::vector2<T, min_allocator<T> > C;
    C c1;
    C c2;
    TEST_LIBCPP_ASSERT_FAILURE(c1.begin() < c2.begin(), "Attempted to compare incomparable iterators");
  }

  return 0;
}
