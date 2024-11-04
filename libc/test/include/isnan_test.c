//===-- Unittests for isnan macro -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef isnan
#error "isnan macro is not defined"
#else
int main(void) {
  assert(!isnan(1.0f));
  assert(!isnan(1.0));
  assert(!isnan(1.0L));
  return 0;
}
#endif
