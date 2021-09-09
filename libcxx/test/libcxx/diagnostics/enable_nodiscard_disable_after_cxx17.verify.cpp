// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test that _LIBCPP_DISABLE_NODISCARD_EXT only disables _LIBCPP_NODISCARD_EXT
// and not _LIBCPP_NODISCARD_AFTER_CXX17.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_NODISCARD
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_NODISCARD_AFTER_CXX17
#include <__config>


_LIBCPP_NODISCARD_EXT int foo() { return 42; }
_LIBCPP_NODISCARD_AFTER_CXX17 int bar() { return 42; }

int main(int, char**) {
  foo(); // expected-warning-re{{ignoring return value of function declared with {{'nodiscard'|warn_unused_result}} attribute}}
  bar(); // OK.
  (void)foo(); // OK.

  return 0;
}
