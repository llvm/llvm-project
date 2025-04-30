/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements LIB3F putc function.  
 */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(PUTC, putc)(DCHAR(ch) DCLEN(ch))
{
  FILE *f;
  int c;
  char *ch = CADR(ch);

  /* DON'T issue any error messages */

  f = __getfile3f(6);
  if (f) {
    c = *ch;
    if (c == fputc(c, f))
      return 0;
    return __io_errno();
  }

  return 0;
}
