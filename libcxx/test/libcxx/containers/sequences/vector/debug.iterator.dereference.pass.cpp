//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Dereference non-dereferenceable iterator.

// UNSUPPORTED: libcxx-no-debug-mode, c++03, windows
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <vector>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector<T> C;
    C c(1);
    C::iterator i = c.end();
    TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable iterator");
  }

  {
    typedef int T;
    typedef std::vector<T, min_allocator<T> > C;
    C c(1);
    C::iterator i = c.end();
    TEST_LIBCPP_ASSERT_FAILURE(*i, "Attempted to dereference a non-dereferenceable iterator");
  }

  return 0;
}
