/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	etime3f.c - Implements LIB3F etime subprogram.  */

#include "ent3f.h"

/* assumes the Unix times system call */

/* Not implemented for WINNT */

#if !defined(_WIN64)
#include <unistd.h>
#include <sys/times.h>
#endif
#define _LIBC_LIMITS_H_
#include <sys/types.h>
#include <limits.h>


#if defined(_WIN64)
   #include "wintimes.h"
   #define CLK_TCK 10000000.0
#else
   #ifndef CLK_TCK
   #define CLK_TCK sysconf(_SC_CLK_TCK)
   #endif
#endif

float ENT3F(ETIME, etime)(float *tarray)
{
  struct tms b;
  float inv_ticks = 1 / (float)CLK_TCK;

  times(&b);
  tarray[0] = ((float)b.tms_utime) * inv_ticks;
  tarray[1] = ((float)b.tms_stime) * inv_ticks;
  return (tarray[0] + tarray[1]);
}

