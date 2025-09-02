//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Basic smoke test for `__log_hardening_failure`.
//
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-has-no-experimental-hardening-observe-semantic

#include <__log_hardening_failure>

#include "test_macros.h"

ASSERT_NOEXCEPT(std::__log_hardening_failure(""));

int main(int, char**) {
  std::__log_hardening_failure("Some message");
  // It's difficult to properly test platform-specific logging behavior of the function; just make sure it exists and
  // can be called at runtime.

  return 0;
}
