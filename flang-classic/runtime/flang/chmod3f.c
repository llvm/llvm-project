/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	chmod3f.c - Implements LIB3F chmod subprogram.  */

/* must include ent3f.h AFTER io3f.h */
/* for chmod */
#include <sys/types.h>
#include <sys/stat.h>

#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#if defined(_WIN64)
#define chmod _chmod
#endif

int ENT3F(CHMOD, chmod)(DCHAR(nam), int *mode DCLEN(nam))
{
  char *p;
  int i;

  p = __fstr2cstr(CADR(nam), CLEN(nam));
  if ((i = chmod(p, *mode)))
    i = __io_errno();
  __cstr_free(p);
  return i;
}
