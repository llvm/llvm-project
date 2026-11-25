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
#error                                                                         \
    "LIBC_HAS_ANNEX_K should not be define without defining __STDC_WANT_LIB_EXT1__"
#endif

// defining LIBC_HAS_ANNEX_K after stdio.h includes annex-k-macros.h
#define __STDC_WANT_LIB_EXT1__ 1

// including strlen_s and rsize_t
#include <string.h>

// including constraint_handler_t and errno_t through a different header file
#include <stdlib.h>

int main(void) {
  // Verify declaration avalibility without referencing strnlen_s at link time
  (void)sizeof(strnlen_s("abc", 2));
  (void)sizeof((errno_t)0);
  (void)sizeof((rsize_t)0);
  (void)sizeof((constraint_handler_t)0);
  return 0;
}
