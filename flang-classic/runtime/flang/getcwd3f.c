/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	getcwd3f.c - Implements LIB3F getcwd subprogram.  */

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

int ENT3F(GETCWD, getcwd)(DCHAR(dir) DCLEN(dir))
{
  char *p;
  int i;

  p = GETCWDM(NULL, CLEN(dir) + 1);
  if (p) {
    __fcp_cstr(CADR(dir), CLEN(dir), p);
    _mp_free(p);
    i = 0;
  } else
    i = __io_errno();

  return i;
}
