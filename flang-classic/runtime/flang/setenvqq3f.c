/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	setenvqq3f.c - Implements DFLIB setenvqq subprogram.  */

#include <stdlib.h>
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(SETENVQQ, setenvqq)(DCHAR(str) DCLEN(str))
{
  int i;
  char *p;

  p = __fstr2cstr(CADR(str), CLEN(str));
  i = putenv(p);
  /* note - putenv stashes the pointer rather than copying the
   *        value, so can't free p.
  __cstr_free(p);
   */

  if (i == -1)
    return 0; /* .false. */
  else
    return -1; /* .true. */
}
