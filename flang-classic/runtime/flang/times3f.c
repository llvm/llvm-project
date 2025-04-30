/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	times3f.c - Implements LIB3F times subprogram.  */

#if !defined(_WIN64)

#include <sys/times.h>
#include "io3f.h"
#include "ent3f.h"

int ENT3F(TIMES, times)(int *buff)
{
  int i;
  struct tms *tbuff = (struct tms *)buff;

  i = times(tbuff);
  if (i == -1)
    i = -__io_errno();

  return i;
}

#endif /* !_WIN64 */
