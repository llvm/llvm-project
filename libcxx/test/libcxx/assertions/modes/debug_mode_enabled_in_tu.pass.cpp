//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can enable the debug mode on a per-TU basis.

// Other hardening modes would additionally trigger the error that they are mutually exclusive.
// REQUIRES: libcpp-hardening-mode=unchecked
// `check_assertion.h` is only available starting from C++11 and requires Unix headers.
// UNSUPPORTED: c++03, !has-unix-headers
// The ability to set a custom abort message is required to compare the assertion message.
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_DEBUG_MODE=1

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE([] {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Should fire");
  }(), "Should fire");

  return 0;
}
