//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that we can override whether assertions are enabled regardless of the hardening mode in use.

// UNSUPPORTED: !libcpp-has-debug-mode
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_ASSERTIONS=0

#include <cassert>

int main(int, char**) {
  _LIBCPP_ASSERT_UNCATEGORIZED(true, "Should not fire");
  _LIBCPP_ASSERT_UNCATEGORIZED(false, "Also should not fire");

  return 0;
}
