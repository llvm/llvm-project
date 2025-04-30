/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getenvqq3f.c - Implements DFLIB getenvqq subprogram.  */

#include <string.h>
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(GETENVQQ, getenvqq)(DCHAR(en), DCHAR(ev) DCLEN(en) DCLEN(ev))
{
  char *p, *q;
  int i;

  q = __fstr2cstr(CADR(en), CLEN(en));
  p = getenv(q);
  if (p == NULL)
    i = 0;
  else
    i = strlen(p);
  __fcp_cstr(CADR(ev), CLEN(ev), p);
  __cstr_free(q);
  return i;
}
