//===-- Unittests for issignaling macro -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
int issignaling(double);
int issignalingf(float);
int issignalingl(long double);

#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef issignaling
#error "issignaling macro is not defined"
#else
int main(void) {
  assert(issignaling(__builtin_nans("")) == 1);
  assert(issignaling(__builtin_nansf("")) == 1);
  assert(issignaling(__builtin_nansl("")) == 1);
  assert(issignaling(1.819f) == 0);
  assert(issignaling(-1.726) == 0);
  assert(issignaling(1.426L) == 0);
  return 0;
}
#endif
