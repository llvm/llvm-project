/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	access3f.c - Implements LIB3F access subroutine.  */

/* must include ent3f.h AFTER io3f.h */
#if !defined(_WIN64)
#include <unistd.h>
#endif

#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(ACCESS, access)(DCHAR(fil), DCHAR(mode) DCLEN(fil) DCLEN(mode))
{
  char *nam;
  int i;
  int stat;
  int m;
  char *mode = CADR(mode);
  int mode_l = CLEN(mode);

  nam = __fstr2cstr(CADR(fil), CLEN(fil));
  m = 0;
  while (mode_l-- > 0) {
    switch (*mode) {
    case 'r':
      m |= 4;
      break;
    case 'w':
      m |= 2;
      break;
    case 'x':
      m |= 1;
      break;
    case ' ':
      break;
    default:
      fprintf(__io_stderr(), "Illegal access mode %c\n", *mode);
    }
    mode++;
  }
  if ((i = access(nam, m)) == 0)
    stat = 0;
  else if (i == -1)
    stat = __io_errno();
  else
    stat = -1;
  __cstr_free(nam);
  return stat;
}
