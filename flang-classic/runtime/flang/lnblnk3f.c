/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	lnblnk3f.c - Implements LIB3F lnblnk subprogram.  */

#include "ent3f.h"

int ENT3F(LNBLNK, lnblnk)(DCHAR(a1) DCLEN(a1))
{
  int i;
  char *a1 = CADR(a1);
  int len = CLEN(a1);

  for (i = len - 1; i >= 0; i--)
    if (a1[i] != ' ')
      return i + 1;
  return 0;
}
