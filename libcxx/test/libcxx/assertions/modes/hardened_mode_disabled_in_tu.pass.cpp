//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can disable the hardened mode on a per-TU basis regardless of how the library was built.

// UNSUPPORTED: libcpp-has-debug-mode
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_HARDENED_MODE=0

#include <cassert>

int main(int, char**) {
  _LIBCPP_ASSERT_UNCATEGORIZED(true, "Should not fire");
  _LIBCPP_ASSERT_UNCATEGORIZED(false, "Also should not fire");

  return 0;
}
