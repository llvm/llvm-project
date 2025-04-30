/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	system3fqq.c - Implements DFLIB systemqq subprogram.  */

#include <stdlib.h>
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(SYSTEMQQ, systemqq)(DCHAR(str) DCLEN(str))
{
  char *p;
  int i;

  p = __fstr2cstr(CADR(str), CLEN(str));
  i = system(p);
  __cstr_free(p);
  if (i == -1)
    return 0; /* .false. */
  else
    return -1; /* .true. */
}
