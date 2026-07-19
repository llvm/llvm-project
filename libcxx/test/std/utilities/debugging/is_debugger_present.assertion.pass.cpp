//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: linux && no-filesystem
// REQUIRES: libcpp-hardening-mode=debug

// <debugging>

// bool is_debugger_present() noexcept;

#include <cassert>
#include <concepts>
#include <debugging>

#include "check_assertion.h"

// Test without a debugger.

void test() {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::is_debugger_present(),
      "Function is not available. Could not open '/proc/self/status' for reading, libc++ was "
      "compiled with _LIBCPP_HAS_NO_FILESYSTEM.");
}

int main(int, char**) {
  test();

  return 0;
}
