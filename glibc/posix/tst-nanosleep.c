/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>


/* Test that nanosleep() does sleep.  */
static int
do_test (void)
{
  /* Current time.  */
  struct timeval tv1;
  (void) gettimeofday (&tv1, NULL);

  struct timespec ts;
  ts.tv_sec = 1;
  ts.tv_nsec = 0;
  TEMP_FAILURE_RETRY (nanosleep (&ts, &ts));

  /* At least one second must have passed.  */
  struct timeval tv2;
  (void) gettimeofday (&tv2, NULL);

  tv2.tv_sec -= tv1.tv_sec;
  tv2.tv_usec -= tv1.tv_usec;
  if (tv2.tv_usec < 0)
    --tv2.tv_sec;

  if (tv2.tv_sec < 1)
    {
      puts ("nanosleep didn't sleep long enough");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
