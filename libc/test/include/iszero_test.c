//===-- Unittests for iszero macro ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef iszero
#error "iszero macro is not defined"
#else
int main(void) {
  assert(iszero(1.0f) == 0);
  assert(iszero(1.0) == 0);
  assert(iszero(1.0L) == 0);
  assert(iszero(0.0f) == 1);
  assert(iszero(0.0) == 1);
  assert(iszero(0.0L) == 1);
  return 0;
}
#endif
