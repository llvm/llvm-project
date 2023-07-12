//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks that if no hardening mode is defined (i.e., in the unchecked mode), by default assertions aren't
// triggered.

// UNSUPPORTED: libcpp-has-hardened-mode, libcpp-has-debug-mode

#include <cassert>

int main(int, char**) {
  // TODO(hardening): remove the `#if` guard once `_LIBCPP_ENABLE_ASSERTIONS` no longer affects hardening modes.
#if !_LIBCPP_ENABLE_ASSERTIONS
  _LIBCPP_ASSERT_UNCATEGORIZED(true, "Should not fire");
  _LIBCPP_ASSERT_UNCATEGORIZED(false, "Also should not fire");
#endif

  return 0;
}
