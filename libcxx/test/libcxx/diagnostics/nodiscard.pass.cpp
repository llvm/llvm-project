//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_NODISCARD_EXT is not defined to [[nodiscard]] unless
// explicitly enabled by _LIBCPP_ENABLE_NODISCARD

#include <__config>

#include "test_macros.h"

_LIBCPP_NODISCARD_EXT int foo() { return 42; }

int main(int, char**) {
  foo(); // OK.

  return 0;
}
