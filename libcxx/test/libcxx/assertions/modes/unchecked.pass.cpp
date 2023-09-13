//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks that if no hardening mode is defined (i.e., in the unchecked mode), by default assertions aren't
// triggered.

// REQUIRES: libcpp-hardening-mode=unchecked

#include <cassert>

bool executed_condition = false;
bool f() { executed_condition = true; return false; }

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Also should not fire");
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(f(), "Should not execute anything");
  assert(!executed_condition); // Really make sure we did not execute anything.

  return 0;
}
