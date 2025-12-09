//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can override the assertion semantic used by any checked hardening mode with `quick-enforce`
// on a per-TU basis (this is valid for the `fast` and `extensive` modes as well, though a no-op).

// `check_assertion.h` is only available starting from C++11 and requires Unix headers and regex support.
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, no-localization
// UNSUPPORTED: libcpp-hardening-mode=none, libcpp-has-no-experimental-hardening-observe-semantic
// ADDITIONAL_COMPILE_FLAGS: -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE(
      [] { _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Should fire without logging a message"); }(),
      "The message should not matter");

  return 0;
}
