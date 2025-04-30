/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	fork3f.c - Implements LIB3F fork subprogram.  */

#if !defined(_WIN64)

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"

int ENT3F(FORK, fork)()
{
  void *f, *q;
  int pid;

  for (f = GET_FIO_FCBS; f != NULL; f = q) {
    q = FIO_FCB_NEXT(f);
    if (fflush(FIO_FCB_FP(f)) != 0)
      return -__io_errno();
  }

  pid = fork();
  if (pid < 0)
    return -__io_errno();
  else
    return pid;
}

#endif /* !_WIN64 */
