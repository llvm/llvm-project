/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Millisecond CPU stopwatch for internal timing
 *
 *  Return the elapsed user+system CPU time in milliseconds
 *  since the most recent call.  Very much not thread-safe.
 */

#ifndef _WIN64

#include <sys/times.h>
#include <unistd.h>
#include "scutil.h"

unsigned long
get_rutime(void)
{
  static long ticks_per_second = -1;
  static unsigned long last = 0;

  struct tms tms;
  unsigned long now, elapsed;

  /* Initialize ticks_per_second. */
#ifdef _SC_CLK_TCK
  if (ticks_per_second <= 0)
    ticks_per_second = sysconf(_SC_CLK_TCK);
#endif /* _SC_CLK_TCK */
  if (ticks_per_second <= 0)
    ticks_per_second = 60; /* a traditional UNIX "jiffy" */

  times(&tms);
  now = tms.tms_utime + tms.tms_stime;
  now *= 1000; /* milliseconds */
  now /= ticks_per_second;

  elapsed = now - last;
  last = now;
  return elapsed;
}

#else

#include <Windows.h>

unsigned long
get_rutime(void)
{
  LARGE_INTEGER ticks_per_second = {-1};
  LARGE_INTEGER ticks;

  unsigned long last = 0;
  unsigned long now, elapsed;

  /* Initialize ticks_per_second. */
  if (ticks_per_second.QuadPart <= 0)
      QueryPerformanceFrequency(&ticks_per_second.QuadPart);

  QueryPerformanceCounter(&ticks);
  now = ticks.QuadPart;
  now *= 1000; /* milliseconds */
  now /= ticks_per_second.QuadPart;

  elapsed = now - last;
  last = now;
  return elapsed;
}

#endif
