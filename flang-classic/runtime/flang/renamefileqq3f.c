/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	renamefileqq3f.c - Implements DFLIB renamefileqq routine.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(RENAMEFILEQQ, renamefileqq)(DCHAR(from),
                                      DCHAR(to) DCLEN(from) DCLEN(to))
{
  int i;

  char *old, *new;

  old = __fstr2cstr(CADR(from), CLEN(from));
  new = __fstr2cstr(CADR(to), CLEN(to));
  i = rename(old, new);
  __cstr_free(old);
  __cstr_free(new);

  if (i == 0)  /* success */
    return -1; /* .true. */
  else         /* failure */
    return 0;  /* .false */
}
