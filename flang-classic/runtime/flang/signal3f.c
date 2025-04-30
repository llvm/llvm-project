/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	signal3f.c - Implements LIB3F signal subprogram.  */

#include <signal.h>

#include "io3f.h"
#include "ent3f.h"

/*
extern void (*signal(int, void (*)(int)))(int);
*/

int ENT3F(SIGNAL, signal)(int *signum, void (*proc)(int), int *flag)
{
  void (*p)();

  if (*flag < 0)
    p = (void (*)())signal(*signum, proc);
  else
    p = (void (*)())signal(*signum, (void (*)())(long)*flag);
  if (p == (void (*)()) - 1)
    return -__io_errno();

  return 0;
}
