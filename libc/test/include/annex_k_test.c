//===-- Tests for Annex K feature selection -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for https://github.com/llvm/llvm-project/issues/195276.
#include <stdio.h>
// Also exercise a type header that stdio.h does not include.
#include "include/llvm-libc-types/constraint_handler_t.h"

#ifdef LIBC_HAS_ANNEX_K
#error "Annex K should not be enabled without __STDC_WANT_LIB_EXT1__"
#endif

// Enable Annex K after stdio.h has already included annex-k-macros.h.
#define __STDC_WANT_LIB_EXT1__ 1
#include <string.h>

// Revisit the shared constraint handler type through a different public header.
#include <stdlib.h>

int main(void) {
  // Verify declaration availability without referencing strnlen_s at link time.
  (void)sizeof(strnlen_s("abc", 2));
  (void)sizeof((errno_t)0);
  (void)sizeof((rsize_t)0);
  (void)sizeof((constraint_handler_t)0);
  return 0;
}
