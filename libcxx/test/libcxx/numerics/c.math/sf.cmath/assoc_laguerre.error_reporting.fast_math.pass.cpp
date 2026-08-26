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

// ADDITIONAL_COMPILE_FLAGS: -ffast-math -Wno-nan-infinity-disabled

// Same as assoc_laguerre.error_reporting.pass.cpp, but compiled with -ffast-math, which
// makes glibc define math_errhandling as 0. Both channels must still be reported by the
// library.
//
// math_errhandling 0 cannot be had without -ffinite-math-only: glibc keys it on
// __FAST_MATH__, which clang defines only when finite-math-only is on as well. So
// error_reporting.h drops the value assertions and the infinite argument here, and
// -Wnan-infinity-disabled has to be silenced -- it fires inside <limits> and <cmath>
// themselves (numeric_limits<T>::infinity(), the isnan/isinf/isfinite implementations, the
// unrelated std::__hermite), which reach this TU because the test suite compiles libc++'s
// headers as non-system headers. None of those sites runs from here; the test body itself no
// longer uses an infinity. See error_reporting.h.

#include <cmath>

#include "error_reporting.h"

int main(int, char**) {
  test_error_reporting();

  return 0;
}
