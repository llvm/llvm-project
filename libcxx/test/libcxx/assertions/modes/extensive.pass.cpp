//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that assertions trigger without the user having to do anything when the extensive hardening mode
// has been enabled by default.

// REQUIRES: libcpp-hardening-mode=extensive
// `check_assertion.h` is only available starting from C++11.
// UNSUPPORTED: c++03
// `check_assertion.h` requires Unix headers.
// REQUIRES: has-unix-headers

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE([] { _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Should fire"); }(), "Should fire");

  return 0;
}
