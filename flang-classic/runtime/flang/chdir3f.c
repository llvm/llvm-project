/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	chdir3f.c - Implements LIB3F chdir subprogram.  */

/* must include ent3f.h AFTER io3f.h */
#if !defined(_WIN64)
#include <unistd.h>
#else
#include <direct.h>
#endif
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

#if defined(_WIN64)
#define chdir _chdir
#endif

int ENT3F(CHDIR, chdir)(DCHAR(path) DCLEN(path))
{
  char *p;
  int i;

  p = __fstr2cstr(CADR(path), CLEN(path));
  if ((i = chdir(p)))
    i = __io_errno();
  __cstr_free(p);
  return i;
}
