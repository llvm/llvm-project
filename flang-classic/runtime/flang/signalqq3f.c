/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	signalqq3f.c - Implements DFLIB signalqq subprogram.  */

#include <signal.h>

#include "io3f.h"
#include "ent3f.h"

#if defined(_WIN64)

#define LONGINTSIZE unsigned long long

LONGINTSIZE
ENT3F(SIGNALQQ, signalqq)(short *signum, void (*proc)(int))
{
  void (*p)();

  p = (void (*)())signal(*signum, proc);
  if (p == (void (*)()) - 1)
    return -1;
  else
    return (LONGINTSIZE)p;
}

#endif /* _WIN64 */
