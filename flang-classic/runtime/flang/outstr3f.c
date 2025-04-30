/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * Implements LIB3F outstr subprogram. 
 */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(OUTSTR, outstr)(DCHAR(ch) DCLEN(ch))
{
  char *ch = CADR(ch);
  int len = CLEN(ch);
  FILE *f;
  int c;

  /* DON'T issue any error messages */

  f = __getfile3f(6);
  if (f) {
    while (len-- >= 0) {
      c = *ch;
      if (c != fputc(c, f))
        return __io_errno();
    }
  }

  return 0;
}
