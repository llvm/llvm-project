/* Test program for POSIX clock_* functions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>


/* We want to see output immediately.  */
#define STDOUT_UNBUFFERED

static int
clock_test (clockid_t cl)
{
  struct timespec old_ts;
  struct timespec ts;
  struct timespec waitit;
  int result = 0;
  int i;

  memset (&ts, '\0', sizeof ts);

  waitit.tv_sec = 0;
  waitit.tv_nsec = 500000000;

  /* Get and print resolution of the clock.  */
  if (clock_getres (cl, &ts) == 0)
    {
      if (ts.tv_nsec < 0 || ts.tv_nsec >= 1000000000)
	{
	  printf ("clock %d: nanosecond value of resolution wrong\n", cl);
	  result = 1;
	}
      else
	printf ("clock %d: resolution = %jd.%09jd secs\n",
		cl, (intmax_t) ts.tv_sec, (intmax_t) ts.tv_nsec);
    }
  else
    {
      printf ("clock %d: cannot get resolution\n", cl);
      result = 1;
    }

  memset (&ts, '\0', sizeof ts);
  memset (&old_ts, '\0', sizeof old_ts);

  /* Next get the current time value a few times.  */
  for (i = 0; i < 10; ++i)
    {
      if (clock_gettime (cl, &ts) == 0)
	{
	  if (ts.tv_nsec < 0 || ts.tv_nsec >= 1000000000)
	    {
	      printf ("clock %d: nanosecond value of time wrong (try %d)\n",
		      cl, i);
	      result = 1;
	    }
	  else
	    {
	      printf ("clock %d: time = %jd.%09jd secs\n",
		      cl, (intmax_t) ts.tv_sec, (intmax_t) ts.tv_nsec);

	      if (memcmp (&ts, &old_ts, sizeof ts) == 0)
		{
		  printf ("clock %d: time hasn't changed (try %d)\n", cl, i);
		  result = 1;

		  old_ts = ts;
		}
	    }
	}
      else
	{
	  printf ("clock %d: cannot get time (try %d)\n", cl, i);
	  result = 1;
	}

      /* Wait a bit before the next iteration.  */
      nanosleep (&waitit, NULL);
    }

  return result;
}

static int
do_test (void)
{
  clockid_t cl;
  int result;

  result = clock_test (CLOCK_REALTIME);

  if (clock_getcpuclockid (0, &cl) == 0)
    /* XXX It's not yet a bug when this fails.  */
    clock_test (cl);
  else
	  printf("CPU clock unavailble, skipping test\n");

  return result;
}
#define TEST_FUNCTION do_test ()


#include "../test-skeleton.c"
