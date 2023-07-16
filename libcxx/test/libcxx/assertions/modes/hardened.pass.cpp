//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that assertions trigger without the user having to do anything when the hardened mode has been
// enabled by default.

// UNSUPPORTED: !libcpp-has-hardened-mode
// `check_assertion.h` is only available starting from C++11.
// UNSUPPORTED: c++03
// `check_assertion.h` requires Unix headers.
// REQUIRES: has-unix-headers
// XFAIL: availability-verbose_abort-missing

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_UNCATEGORIZED(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE([] {
    _LIBCPP_ASSERT_UNCATEGORIZED(false, "Should fire");
  }(), "Should fire");

  return 0;
}
