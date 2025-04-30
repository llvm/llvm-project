/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	fdate3f.c - Implements LIB3F fdate subprogram.  */

#include "ent3f.h"

#include <time.h>
#include "utils3f.h"

#if !defined(_WIN64)
WIN_MSVCRT_IMP char *WIN_CDECL ctime(const time_t *);
#endif

void ENT3F(FDATE, fdate)(DCHAR(str) DCLEN(str))
{
  char *str = CADR(str);
  int len = CLEN(str);
  time_t t;
  char *p;
  int i;

  t = time(0);
  p = ctime(&t);
  __fcp_cstr(str, len, p);
  for (i = len - 1; i >= 0; i--)
    if (str[i] == '\n') {
      str[i] = ' ';
      break;
    }

  return;
}
