//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can override any hardening mode with the extensive hardening mode on a per-TU basis.

// `check_assertion.h` is only available starting from C++11 and requires Unix headers and regex support.
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, no-localization
// The ability to set a custom abort message is required to compare the assertion message (which only happens in the
// debug mode).
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing
// HWASAN replaces TRAP with abort or error exit code.
// XFAIL: hwasan
// ADDITIONAL_COMPILE_FLAGS: -U_LIBCPP_HARDENING_MODE -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_COMPATIBLE_ALLOCATOR(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE(
      [] { _LIBCPP_ASSERT_COMPATIBLE_ALLOCATOR(false, "Extensive-mode assertions should fire"); }(),
      "Extensive-mode assertions should fire");
  _LIBCPP_ASSERT_INTERNAL(false, "Debug-mode assertions should not fire");

  return 0;
}
