//===-- Unittests for signbit macro ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "include/llvm-libc-macros/math-function-macros.h"

#include <assert.h>

// check if macro is defined
#ifndef signbit
#error "signbit macro is not defined"
#else
int main(void) {
  assert(!signbit(1.0f));
  assert(!signbit(1.0));
  assert(!signbit(1.0L));
  assert(signbit(-1.0f));
  assert(signbit(-1.0));
  assert(signbit(-1.0L));
  return 0;
}
#endif
