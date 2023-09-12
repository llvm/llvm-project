//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can disable the debug mode on a per-TU basis.

// Other hardening modes would still make the assertions fire (disabling the debug mode doesn't disable e.g. the
// hardened mode).
// REQUIRES: libcpp-hardening-mode=debug
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_DEBUG_MODE=0

#include <cassert>

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Also should not fire");

  return 0;
}
