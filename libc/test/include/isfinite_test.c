//===-- Unittests for isfinite macro --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef isfinite
#error "isfinite macro is not defined"
#else
int main(void) {
  assert(isfinite(1.0f));
  assert(isfinite(1.0));
  assert(isfinite(1.0L));
  return 0;
}
#endif
