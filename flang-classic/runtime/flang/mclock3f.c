/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	mclock3f.c - Implements LIB3F mclock subprogram.  */
#include "ent3f.h"

/* assumes the Unix times system call */

#if   defined(_WIN64)

#include <time.h>

int ENT3F(MCLOCK, mclock)(void) { return clock(); }

#else
#include <sys/times.h>

int ENT3F(MCLOCK, mclock)(void)
{
  struct tms buffer;

  times(&buffer);
  return (buffer.tms_utime + buffer.tms_cutime + buffer.tms_cstime);
}

#endif
