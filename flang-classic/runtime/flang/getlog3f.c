/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getlog3f.c - Implements LIB3F getlog subprogram.  */

#if !defined(_WIN64)

#include "ent3f.h"
#include "utils3f.h"

extern char *getlogin();

void ENT3F(GETLOG, getlog)(DCHAR(nm) DCLEN(nm))
{
  char *p;

  p = getlogin();
  __fcp_cstr(CADR(nm), CLEN(nm), p);
}

#endif /* !_WIN64 */
