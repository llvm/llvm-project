//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO(hardening): remove in LLVM 19.
// This test ensures that enabling assertions now enables the hardened mode.

// `check_assertion.h` is only available starting from C++11 and requires Unix headers.
// UNSUPPORTED: c++03, !has-unix-headers
// The ability to set a custom abort message is required to compare the assertion message.
// XFAIL: availability-verbose_abort-missing
// Debug mode is mutually exclusive with hardened mode.
// UNSUPPORTED: libcpp-hardening-mode=debug
// Ignore the warning about `_LIBCPP_ENABLE_ASSERTIONS` being deprecated.
// ADDITIONAL_COMPILE_FLAGS: -Wno-error -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  static_assert(_LIBCPP_ENABLE_HARDENED_MODE == 1, "Hardened mode should be implicitly enabled");

  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE([] {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Should fire");
  }(), "Should fire");

  return 0;
}
