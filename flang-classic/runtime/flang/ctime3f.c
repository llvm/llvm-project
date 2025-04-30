/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	ctime3f.c - Implements LIB3F ctime subprogram.  */

#include "ent3f.h"
#include "utils3f.h"

extern char *ctime(long *);

/* ctime is a character function */

static void
ctime_c(char *tm, int tml, long stime)
{
  char *p;
  int i;

  p = ctime(&stime); /* ctime arg is 'pointer to' */
  __fcp_cstr(tm, tml, p);
  for (i = tml - 1; i >= 0; i--)
    if (tm[i] == '\n') {
      tm[i] = ' ';
      break;
    }

  return;
}

void ENT3F(CTIME, ctime)(DCHAR(tm) DCLEN(tm), int *stime)
{
  ctime_c(CADR(tm), CLEN(tm), (long)(*stime));
}

void ENT3F(CTIME8, ctime8)(DCHAR(tm) DCLEN(tm), long long *stime)
{
  ctime_c(CADR(tm), CLEN(tm), (long)*stime);
}
