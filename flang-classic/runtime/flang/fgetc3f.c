/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * Implements LIB3F fgetc function.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(FGETC, fgetc)(int *lu, DCHAR(ch) DCLEN(ch))
{
  FILE *f;
  int c;
  char *ch = CADR(ch);

  /*** DON'T issue any error messages ***/

  f = __getfile3f(*lu);
  if (f) {
    c = fgetc(f);
    if (c == EOF) {
      if (__io_feof(f))
        return -1;
      return __io_errno();
    }
    *ch = c;
  }

  return 0;
}
