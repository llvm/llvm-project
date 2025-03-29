//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Decrement iterator prior to begin.

// REQUIRES: has-unix-headers, libcpp-has-abi-bounded-iterators-in-vector
// UNSUPPORTED: libcpp-hardening-mode=none, c++03

#include <vector>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector<T> C;
    C c(1);
    C::iterator i = c.end();
    --i;
    assert(i == c.begin());
    TEST_LIBCPP_ASSERT_FAILURE(--i, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
  }

  {
    typedef int T;
    typedef std::vector<T, min_allocator<T> > C;
    C c(1);
    C::iterator i = c.end();
    --i;
    assert(i == c.begin());
    TEST_LIBCPP_ASSERT_FAILURE(--i, "__bounded_iter::operator--: Attempt to rewind an iterator past the start");
  }

  return 0;
}
