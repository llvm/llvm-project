//===-- Unittests for issubnormal macro -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef issubnormal
#error "issubnormal macro is not defined"
#else
int main(void) {
  assert(issubnormal(1.819f) == 0);
  assert(issubnormal(-1.726) == 0);
  assert(issubnormal(1.426L) == 0);
  assert(issubnormal(1e-308) == 1);
  assert(issubnormal(-1e-308) == 1);
  return 0;
}
#endif
