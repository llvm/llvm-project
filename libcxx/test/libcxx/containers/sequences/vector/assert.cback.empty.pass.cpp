//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Call back() on empty const container.

// UNSUPPORTED: c++03, windows, libcxx-no-debug-mode
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=0

#include <vector>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector<T, min_allocator<T> > C;
    const C c;
    TEST_LIBCPP_ASSERT_FAILURE(c.back(), "back() called on an empty vector");
  }

  {
    typedef int T;
    typedef std::vector<T> C;
    const C c;
    TEST_LIBCPP_ASSERT_FAILURE(c.back(), "back() called on an empty vector");
  }

  return 0;
}
