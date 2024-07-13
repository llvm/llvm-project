//===-- Unittests for isfinite macro -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * The intent of this test is validate that:
 * 1. We provide the definition of the various type generic macros of stdbit.h
 * (the macros are transitively included from stdbit-macros.h by stdbit.h).
 * 2. It dispatches to the correct underlying function.
 * Because unit tests build without public packaging, the object files produced
 * do not contain non-namespaced symbols.
 */

/*
 * Declare these BEFORE including stdbit-macros.h so that this test may still be
 * run even if a given target doesn't yet have these individual entrypoints
 * enabled.
 */

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

