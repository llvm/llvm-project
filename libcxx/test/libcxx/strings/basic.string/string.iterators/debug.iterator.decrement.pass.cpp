//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Decrement iterator prior to begin.

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcxx-no-debug-mode, c++03
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

#include <string>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::string C;
    C c(1, '\0');
    C::iterator i = c.end();
    --i;
    assert(i == c.begin());
    TEST_LIBCPP_ASSERT_FAILURE(--i, "Attempted to decrement a non-decrementable iterator");
  }

  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > C;
    C c(1, '\0');
    C::iterator i = c.end();
    --i;
    assert(i == c.begin());
    TEST_LIBCPP_ASSERT_FAILURE(--i, "Attempted to decrement a non-decrementable iterator");
  }

  return 0;
}
