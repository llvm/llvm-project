//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator position) with end()

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: !libcpp-has-hardened-mode && !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
    S l1("123");
    S::const_iterator i = l1.end();
    TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i), "string::erase(iterator) called with a non-dereferenceable iterator");
  }

  {
    std::string l1("123");
    std::string::const_iterator i = l1.end();
    TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i), "string::erase(iterator) called with a non-dereferenceable iterator");
  }

  return 0;
}
