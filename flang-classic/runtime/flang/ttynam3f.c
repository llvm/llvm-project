/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	ttynam3f.c - Implements LIB3F ttynam subprogram.  */

#if !defined(_WIN64)

#include <unistd.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

/* ttynam is a character function */
void ENT3F(TTYNAM, ttynam)(DCHAR(nm) DCLEN(nm), int *lu)
{
  int u;
  char *p;

  switch (*lu) {
  case 0:
    u = 2;
    break;
  case 5:
    u = 0;
    break;
  case 6:
    u = 1;
    break;
  default:
    p = 0;
    goto sk;
  }
  p = ttyname(u);
sk:
  __fcp_cstr(CADR(nm), CLEN(nm), p);
  /*
  if (p) free(p);
  */

  return;
}

#endif /* !_WIN64 */
