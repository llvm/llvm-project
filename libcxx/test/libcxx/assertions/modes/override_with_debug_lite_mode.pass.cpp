//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can override any hardening mode with the debug-lite mode on a per-TU basis.

// `check_assertion.h` is only available starting from C++11.
// UNSUPPORTED: c++03
// `check_assertion.h` requires Unix headers.
// REQUIRES: has-unix-headers
// The ability to set a custom abort message is required to compare the assertion message.
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG_LITE

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_COMPATIBLE_ALLOCATOR(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE([] {
    _LIBCPP_ASSERT_COMPATIBLE_ALLOCATOR(false, "Debug-lite-mode assertions should fire");
  }(), "Debug-lite-mode assertions should fire");
  _LIBCPP_ASSERT_INTERNAL(false, "Debug-mode assertions should not fire");

  return 0;
}
