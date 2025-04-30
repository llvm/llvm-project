/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	pxffileno.c - Implements LIB3F pxffileno subroutine.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"
#include "stdioInterf.h"

void ENT3F(PXFFILENO, pxffileno)(int *lu, int *fd, int *err)
{
  FILE *f;

  f = __getfile3f(*lu);
  if (f == (FILE *)0) {
    *err = FIO_EOPENED;
    return;
  }
  *err = 0;
  fflush(f);
  *fd = __io_getfd(f);
  return;
}
