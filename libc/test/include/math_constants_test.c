//===-- Unittests for math constants --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-macros.h"

#define IS_DOUBLE(X) _Generic((X), double: 1, default: 0)

#define IS_FLOAT(X) _Generic((X), float: 1, default: 0)

// check if macro is defined
#ifndef M_PI
#error "M_PI macro is not defined"
#else
int main(void) {
  _Static_assert(IS_DOUBLE(M_PI), "M_PI is not of double type.");
  _Static_assert(IS_FLOAT(M_PIf), "M_PIf is not of float type.");
  return 0;
}
#endif
