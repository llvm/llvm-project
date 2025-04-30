/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getenv3f.c - Implements LIB3F getenv subprogram.  */

#include "ent3f.h"
#include "utils3f.h"

void ENT3F(GETENV, getenv)(DCHAR(en), DCHAR(ev) DCLEN(en) DCLEN(ev))
{
  char *p, *q;

  q = __fstr2cstr(CADR(en), CLEN(en));
  p = getenv(q);
  __fcp_cstr(CADR(ev), CLEN(ev), p);
  __cstr_free(q);
}
