/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	timef3f.c - Implements timef subprogram.  */

/* assumes the Unix times system call */
/* how do we do this for WINNT */
#include "ent3f.h"

#if !defined(_WIN64)
#define _LIBC_LIMITS_H_
#include <unistd.h>
#include <sys/times.h>
#endif
#include <sys/types.h>
#include <limits.h>

#if defined(_WIN64)
#include "wintimes.h"
#endif

#ifndef CLK_TCK
#define CLK_TCK sysconf(_SC_CLK_TCK)
#endif

static clock_t start = 0;

double ENT3F(TIMEF, timef)(float *tarray)
{
  struct tms b;
  clock_t current;
  double duration;
  double inv_ticks = 1 / (double)CLK_TCK;

  times(&b);
  if (start == 0) {
    start = b.tms_utime + b.tms_stime;
    current = start;
  } else
    current = b.tms_utime + b.tms_stime;

  duration = ((double)(current - start)) * inv_ticks;
  return duration;
}

