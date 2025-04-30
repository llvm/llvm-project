/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	perror3f.c - Implements LIB3F perror subprogram.  */

#include <string.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

void ENT3F(PERROR, perror)(DCHAR(str) DCLEN(str))
{
  FILE *fp;
  char *p;
  char *str = CADR(str);
  int str_l = CLEN(str);

  p = strerror(__io_errno());
  fp = __getfile3f(0);
  if (str_l > 0) {
    do {
      fputc((int)*str, fp);
      str++;
    } while (--str_l > 0);
    fputc(':', fp);
    fputc(' ', fp);
  }
  fprintf(fp, "%s", p);
  if (__PC_DOS)
    fputc('\r', fp);
  fputc('\n', fp);

  return;
}
