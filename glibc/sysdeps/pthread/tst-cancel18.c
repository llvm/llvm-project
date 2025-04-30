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


static pthread_barrier_t b;


/* Cleanup handling test.  */
static int cl_called;

static void
cl (void *arg)
{
  ++cl_called;
}


static void *
tf (void *arg)
{
  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  pthread_cleanup_push (cl, NULL);

  struct timespec ts = { .tv_sec = arg == NULL ? 10000000 : 0, .tv_nsec = 0 };
  TEMP_FAILURE_RETRY (clock_nanosleep (CLOCK_REALTIME, 0, &ts, &ts));

  pthread_cleanup_pop (0);

  puts ("clock_nanosleep returned");

  exit (1);
}


static int
do_test (void)
{
  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("1st create failed");
      return 1;
    }

  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  struct timespec  ts = { .tv_sec = 0, .tv_nsec = 100000000 };
  while (nanosleep (&ts, &ts) != 0)
    continue;

  puts ("going to cancel in-time");
  if (pthread_cancel (th) != 0)
    {
      puts ("1st cancel failed");
      return 1;
    }

  void *status;
  if (pthread_join (th, &status) != 0)
    {
      puts ("1st join failed");
      return 1;
    }
  if (status != PTHREAD_CANCELED)
    {
      puts ("1st thread not canceled");
      return 1;
    }

  if (cl_called == 0)
    {
      puts ("cleanup handler not called");
      return 1;
    }
  if (cl_called > 1)
    {
      puts ("cleanup handler called more than once");
      return 1;
    }

  puts ("in-time cancellation succeeded");


  cl_called = 0;

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("2nd create failed");
      return 1;
    }

  puts ("going to cancel early");
  if (pthread_cancel (th) != 0)
    {
      puts ("2nd cancel failed");
      return 1;
    }

  r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  if (pthread_join (th, &status) != 0)
    {
      puts ("2nd join failed");
      return 1;
    }
  if (status != PTHREAD_CANCELED)
    {
      puts ("2nd thread not canceled");
      return 1;
    }

  if (cl_called == 0)
    {
      printf ("cleanup handler not called\n");
      return 1;
    }
  if (cl_called > 1)
    {
      printf ("cleanup handler called more than once\n");
      return 1;
    }

  puts ("early cancellation succeeded");

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
