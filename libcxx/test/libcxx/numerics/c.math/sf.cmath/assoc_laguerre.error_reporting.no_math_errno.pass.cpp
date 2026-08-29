//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// These functions are implemented in the built library, so a program using them fails to
// load against a back-deployment target whose libc++ predates them.
// XFAIL: availability-mathematical_special_functions-missing

// ADDITIONAL_COMPILE_FLAGS: -fno-math-errno

// Same as assoc_laguerre.error_reporting.pass.cpp, but compiled with -fno-math-errno,
// which makes glibc define math_errhandling as MATH_ERREXCEPT alone. The library still
// has to set errno: it cannot see the caller's flags. See error_reporting.h.

#include <cmath>

#include "error_reporting.h"

int main(int, char**) {
  test_error_reporting();

  return 0;
}
