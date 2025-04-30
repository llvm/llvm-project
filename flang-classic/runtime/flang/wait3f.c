/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	wait3f.c - Implements LIB3F wait subprogram.  */

#if !defined(_WIN64)

#include <sys/types.h>
#include <sys/wait.h>
#include "ent3f.h"

/* The type of the wait system call argument differs between various
 * Linux flavaors and OSX
 */
#define WAIT_STAT int*

int ENT3F(WAIT, wait)(int *st) 
{
  WAIT_STAT wst = (WAIT_STAT)st;
  return wait(wst);
}

#endif /* !_WIN64 */
