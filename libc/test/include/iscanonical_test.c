//===-- Unittests for iscanonical macro -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/float16-macros.h"

int iscanonical(double);
int iscanonicalf(float);
int iscanonicall(long double);
#ifdef LIBC_TYPES_HAS_FLOAT16
int iscanonicalf16(_Float16);
#endif
#if defined(LIBC_TYPES_HAS_FLOAT128) && (LDBL_MANT_DIG != 113)
int iscanonicalf128(float128);
#endif

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
#ifdef LIBC_TYPES_HAS_FLOAT16
  assert(iscanonical(__builtin_nansf16("")) == 0);
  assert(iscanonical((_Float16)1.0) == 1);
#endif
#if defined(LIBC_TYPES_HAS_FLOAT128) && (LDBL_MANT_DIG != 113)
  assert(iscanonical(__builtin_nansf128("")) == 0);
  assert(iscanonical((float128)1.0) == 1);
#endif
  return 0;
}
#endif
