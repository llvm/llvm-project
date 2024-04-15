//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can override any hardening mode with the debug mode on a per-TU basis.

// `check_assertion.h` is only available starting from C++11 and requires Unix headers and regex support.
// UNSUPPORTED: c++03, !has-unix-headers, no-localization
// The ability to set a custom abort message is required to compare the assertion message.
// XFAIL: availability-verbose_abort-missing
// ADDITIONAL_COMPILE_FLAGS: -U_LIBCPP_HARDENING_MODE -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_INTERNAL(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE([] { _LIBCPP_ASSERT_INTERNAL(false, "Debug-mode assertions should fire"); }(),
                             "Debug-mode assertions should fire");

  return 0;
}
