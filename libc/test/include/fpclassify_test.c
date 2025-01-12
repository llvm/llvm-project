//===-- Unittests for fpclassify macro ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef fpclassify
#error "fpclassify macro is not defined"
#else
int main(void) {
  assert(fpclassify(1.819f) == FP_NORMAL);
  assert(fpclassify(-1.726) == FP_NORMAL);
  assert(fpclassify(1.426L) == FP_NORMAL);
  assert(fpclassify(-0.0f) == FP_ZERO);
  assert(fpclassify(0.0) == FP_ZERO);
  assert(fpclassify(-0.0L) == FP_ZERO);
  return 0;
}
#endif
