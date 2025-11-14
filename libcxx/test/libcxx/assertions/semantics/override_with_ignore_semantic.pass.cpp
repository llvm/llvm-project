//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can override the assertion semantic used by any hardening mode with `ignore` on a per-TU
// basis (this is valid for the `none` mode as well, though a no-op).

// `check_assertion.h` is only available starting from C++11 and requires Unix headers and regex support.
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, no-localization
// UNSUPPORTED: libcpp-has-no-experimental-hardening-observe-semantic
// ADDITIONAL_COMPILE_FLAGS: -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_IGNORE

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Also should not fire");

  return 0;
}
