//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_ASSERT doesn't do anything when assertions are disabled.
// We need to use -Wno-macro-redefined because the test suite defines
// _LIBCPP_ENABLE_ASSERTIONS=1 under some configurations.

// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_ASSERTIONS=0

#include <cassert>

bool executed_condition = false;
bool f() { executed_condition = true; return false; }

int main(int, char**) {
  _LIBCPP_ASSERT(f(), "message"); // should not execute anything
  assert(!executed_condition); // really make sure we did not execute anything at all
  return 0;
}
