/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	kill3f.c - Implements LIB3F kill subprogram.  */

#if !defined(_WIN64)

#define POSIX 1
#include <sys/types.h>
#include <signal.h>

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"

int ENT3F(KILL, kill)(int *pid, int *sig)
{
  int i;

  if ((i = kill(*pid, *sig)))
    i = __io_errno();
  return i;
}

#endif /* !_WIN64 */
