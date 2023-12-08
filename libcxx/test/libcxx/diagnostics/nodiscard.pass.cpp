//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_NODISCARD_EXT is not defined to [[nodiscard]] when
// _LIBCPP_DISABLE_NODISCARD_EXT is defined

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_EXT

#include <__config>

_LIBCPP_NODISCARD_EXT int foo() { return 42; }

int main(int, char**) {
  foo(); // OK.

  return 0;
}
