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
  assert(fpclassify(1.819f) == 4);
  assert(fpclassify(-1.726) == 4);
  assert(fpclassify(1.426L) == 4);
  assert(fpclassify(-0.0f) == 2);
  assert(fpclassify(0.0) == 2);
  assert(fpclassify(-0.0L) == 2);
  return 0;
}
#endif
