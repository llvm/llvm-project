/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	unlink3f.c - Implements LIB3F unlink subroutine.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(UNLINK, unlink)(DCHAR(fil) DCLEN(fil))
{
  char *nam;
  int i;

  nam = __fstr2cstr(CADR(fil), CLEN(fil));
  i = unlink(nam);
  __cstr_free(nam);
  if (i == 0)
    return 0;
  return __io_errno();
}
