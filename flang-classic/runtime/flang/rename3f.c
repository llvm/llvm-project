/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	rename3f.c - Implements LIB3F rename subprogram.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(RENAME, rename)(DCHAR(from), DCHAR(to) DCLEN(from) DCLEN(to))
{
  int i;

  char *old, *new;

  old = __fstr2cstr(CADR(from), CLEN(from));
  new = __fstr2cstr(CADR(to), CLEN(to));
  if ((i = rename(old, new)))
    i = __io_errno();
  __cstr_free(old);
  __cstr_free(new);

  return i;
}
