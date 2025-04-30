/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>


#if defined _POSIX_THREAD_CPUTIME && _POSIX_THREAD_CPUTIME >= 0
static pthread_barrier_t b2;
static pthread_barrier_t bN;


static void *
tf (void *arg)
{
  int e = pthread_barrier_wait (&b2);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  e = pthread_barrier_wait (&bN);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  return NULL;
}
#endif


int
do_test (void)
{
#if defined _POSIX_THREAD_CPUTIME && _POSIX_THREAD_CPUTIME >= 0
# define N 10

# if _POSIX_THREAD_CPUTIME == 0
  if (sysconf (_SC_THREAD_CPUTIME) < 0)
    {
      puts ("_POSIX_THREAD_CPUTIME option not available");
      return 0;
    }
# endif

  if (pthread_barrier_init (&b2, NULL, 2) != 0
      || pthread_barrier_init (&bN, NULL, N + 1) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  struct timespec ts = { .tv_sec = 0, .tv_nsec = 100000000 };
  TEMP_FAILURE_RETRY (nanosleep (&ts, &ts));

  pthread_t th[N + 1];
  clockid_t cl[N + 1];
# ifndef CLOCK_THREAD_CPUTIME_ID
  if (pthread_getcpuclockid (pthread_self (), &cl[0]) != 0)
    {
      puts ("own pthread_getcpuclockid failed");
      return 1;
    }
# else
  cl[0] = CLOCK_THREAD_CPUTIME_ID;
# endif

  pthread_attr_t at;

  if (pthread_attr_init (&at) != 0)
    {
      puts ("attr_init failed");
      return 1;
    }

  if (pthread_attr_setstacksize (&at, 1 * 1024 * 1024) != 0)
    {
      puts ("attr_setstacksize failed");
      return 1;
    }

  int i;
  int e;
  for (i = 0; i < N; ++i)
    {
      if (pthread_create (&th[i], &at, tf, NULL) != 0)
	{
	  puts ("create failed");
	  return 1;
	}

      e = pthread_barrier_wait (&b2);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("barrier_wait failed");
	  return 1;
	}

      ts.tv_sec = 0;
      ts.tv_nsec = 100000000;
      TEMP_FAILURE_RETRY (nanosleep (&ts, &ts));

      if (pthread_getcpuclockid (th[i], &cl[i + 1]) != 0)
	{
	  puts ("pthread_getcpuclockid failed");
	  return 1;
	}
    }

  if (pthread_attr_destroy (&at) != 0)
    {
      puts ("attr_destroy failed");
      return 1;
    }

  struct timespec t[N + 1];
  for (i = 0; i < N + 1; ++i)
    if (clock_gettime (cl[i], &t[i]) != 0)
      {
	printf ("clock_gettime round %d failed\n", i);
	return 1;
      }

  for (i = 0; i < N; ++i)
    {
      struct timespec diff;

      diff.tv_sec = t[i].tv_sec - t[i + 1].tv_sec;
      diff.tv_nsec = t[i].tv_nsec - t[i + 1].tv_nsec;
      if (diff.tv_nsec < 0)
	{
	  diff.tv_nsec += 1000000000;
	  --diff.tv_sec;
	}

      if (diff.tv_sec < 0 || (diff.tv_sec == 0 && diff.tv_nsec < 100000000))
	{
	  printf ("\
difference between thread %d and %d too small (%ld.%09ld)\n",
		  i, i + 1, (long int) diff.tv_sec, (long int) diff.tv_nsec);
	  return 1;
	}

      printf ("diff %d->%d: %ld.%09ld\n",
	      i, i + 1, (long int) diff.tv_sec, (long int) diff.tv_nsec);
    }

  ts.tv_sec = 0;
  ts.tv_nsec = 0;
  for (i = 0; i < N + 1; ++i)
    if (clock_settime (cl[i], &ts) != 0)
      {
	printf ("clock_settime(%d) round %d failed\n", cl[i], i);
	return 1;
      }

  for (i = 0; i < N + 1; ++i)
    {
      if (clock_gettime (cl[i], &ts) != 0)
	{
	  puts ("clock_gettime failed");
	  return 1;
	}

      if (ts.tv_sec > t[i].tv_sec
	  || (ts.tv_sec == t[i].tv_sec && ts.tv_nsec > t[i].tv_nsec))
	{
	  puts ("clock_settime didn't reset clock");
	  return 1;
	}
    }
#endif

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
