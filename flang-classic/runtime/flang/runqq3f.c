/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	run3fqq.c - Implements DFLIB runqq subprogram.  */

#include <string.h>
#include "ent3f.h"
#include "mpalloc.h"
#include "utils3f.h"

short ENT3F(RUNQQ, runqq)(DCHAR(fname), DCHAR(cline) DCLEN(fname) DCLEN(cline))
{
  char *fn;
  char *cl;
  char *m;
  short i;
  int len;

  fn = __fstr2cstr(CADR(fname), CLEN(fname));
  cl = __fstr2cstr(CADR(cline), CLEN(cline));
  len = strlen(fn) + strlen(cl) + 1;
  m = (char *)_mp_malloc((len + 1) * sizeof(char));
  m = strcpy(m, fn);
  m = strcat(m, " ");
  m = strcat(m, cl);
  i = system(m);
  _mp_free(m);
  __cstr_free(fn);
  __cstr_free(cl);
  return i;
}
