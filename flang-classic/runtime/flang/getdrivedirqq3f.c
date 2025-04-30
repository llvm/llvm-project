/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getdrivedirqq3f.c - Implements DFLIB getdrivedirqq subprogram.  */

#include <string.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"
#include "mpalloc.h"

#if defined(_WIN64)
#define GETCWDM _getcwd /* getcwd deprecated in Windows in VC 2005 */
#else
#define GETCWDM getcwd
#endif

int ENT3F(GETDRIVEDIRQQ, getdrivedirqq)(DCHAR(dir) DCLEN(dir))
{
  char *p, *q;
  int i, l1, l2;

  q = __fstr2cstr(CADR(dir), CLEN(dir));
  l1 = CLEN(dir) + 1;
  if (strlen(q) + 1 < l1)
    l1 = strlen(q);
  __cstr_free(q);
  p = GETCWDM(NULL, l1);
  if (p) {
    __fcp_cstr(CADR(dir), CLEN(dir), p);
    i = 0;
    l2 = strlen(p);
    if (l2 > CLEN(dir))
      l2 = 0;
    _mp_free(p);
  } else {
    i = __io_errno();
    l2 = 0;
  }

  return l2;
}
