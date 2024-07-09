//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Make sure we catch invalid uses of std::deque::erase(iterator).

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <deque>

#include "check_assertion.h"

int main(int, char**) {
  // With an invalid iterator
  {
    std::deque<int> v = {1, 2, 3, 4, 5};
    TEST_LIBCPP_ASSERT_FAILURE(v.erase(v.end()), "deque::erase(iterator) called with a non-dereferenceable iterator");
  }

  // Note that we currently can't catch misuse by erasing with an iterator from another container.

  return 0;
}
