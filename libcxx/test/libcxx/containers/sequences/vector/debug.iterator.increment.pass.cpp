//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Increment iterator past end.

// REQUIRES: has-unix-headers, libcpp-has-abi-bounded-iterators
// UNSUPPORTED: libcpp-hardening-mode=none, c++03

#include <vector>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

template <typename T, typename A>
void fill_to_capacity(std::vector<T, A>& vec) {
  // Fill vec up to its capacity. Our bounded iterators currently unable to
  // catch accesses between size and capacity due to iterator stability
  // guarantees. This function clears those effects.
  while (vec.size() < vec.capacity()) {
    vec.push_back(T());
  }
}

int main(int, char**) {
  {
    typedef int T;
    typedef std::vector<T> C;
    C c(1);
    fill_to_capacity(c);
    C::iterator i = c.begin();
    i += c.size();
    assert(i == c.end());
    TEST_LIBCPP_ASSERT_FAILURE(++i, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
  }

  {
    typedef int T;
    typedef std::vector<T, min_allocator<T> > C;
    C c(1);
    fill_to_capacity(c);
    C::iterator i = c.begin();
    i += c.size();
    assert(i == c.end());
    TEST_LIBCPP_ASSERT_FAILURE(++i, "__bounded_iter::operator++: Attempt to advance an iterator past the end");
  }

  return 0;
}
