//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can override any hardening mode with the unchecked mode on a per-TU basis.

// `check_assertion.h` is only available starting from C++11 and requires Unix headers and regex support.
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, no-localization
// ADDITIONAL_COMPILE_FLAGS: -U_LIBCPP_HARDENING_MODE -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Also should not fire");

  return 0;
}
