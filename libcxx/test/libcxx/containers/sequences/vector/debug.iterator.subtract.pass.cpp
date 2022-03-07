//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Subtract iterators from different containers.

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <vector>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector<T> C;
    C c1;
    C c2;
    TEST_LIBCPP_ASSERT_FAILURE(c1.begin() - c2.begin(), "Attempted to subtract incompatible iterators");
  }

  {
    typedef int T;
    typedef std::vector<T, min_allocator<T> > C;
    C c1;
    C c2;
    TEST_LIBCPP_ASSERT_FAILURE(c1.begin() - c2.begin(), "Attempted to subtract incompatible iterators");
  }

  return 0;
}
