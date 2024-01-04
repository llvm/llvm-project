//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO(hardening): remove in LLVM 20.
// This test ensures that enabling assertions with the legacy `_LIBCPP_ENABLE_ASSERTIONS` now enables the extensive
// hardening mode.

// `check_assertion.h` is only available starting from C++11 and requires Unix headers.
// UNSUPPORTED: c++03, !has-unix-headers
// The ability to set a custom abort message is required to compare the assertion message.
// XFAIL: availability-verbose_abort-missing
// Note that GCC doesn't support `-Wno-macro-redefined`.
// ADDITIONAL_COMPILE_FLAGS: -U_LIBCPP_HARDENING_MODE -D_LIBCPP_ENABLE_ASSERTIONS=1

#include <cassert>
#include "check_assertion.h"

int main(int, char**) {
  static_assert(_LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_EXTENSIVE,
                "The extensive hardening mode should be implicitly enabled");

  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  TEST_LIBCPP_ASSERT_FAILURE([] { _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Should fire"); }(), "Should fire");

  return 0;
}
