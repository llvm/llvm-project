//===-- Unittests for isnormal macro --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef isnormal
#error "isnormal macro is not defined"
#else
int main(void) {
  assert(isnormal(1.819f) == 1);
  assert(isnormal(-1.726) == 1);
  assert(isnormal(1.426L) == 1);
  assert(isnormal(-0.0f) == 0);
  assert(isnormal(0.0) == 0);
  assert(isnormal(-0.0L) == 0);
  return 0;
}
#endif
