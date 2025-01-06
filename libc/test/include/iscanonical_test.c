//===-- Unittests for iscanonical macro -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
int iscanonical(double);
int iscanonicalf(float);
int iscanonicall(long double);

#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef iscanonical
#error "iscanonical macro is not defined"
#else
int main(void) {
  assert(iscanonical(__builtin_nans("")) == 0);
  assert(iscanonical(__builtin_nansf("")) == 0);
  assert(iscanonical(__builtin_nansl("")) == 0);
  assert(iscanonical(1.819f) == 1);
  assert(iscanonical(-1.726) == 1);
  assert(iscanonical(1.426L) == 1);
  return 0;
}
#endif
