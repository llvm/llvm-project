/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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
#include <unistd.h>

pthread_barrier_t b;
int seen;

static void *
tf (void *arg)
{
  int old;
  int r = pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &old);
  if (r != 0)
    {
      puts ("setcancelstate failed");
      exit (1);
    }

  r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  for (int i = 0; i < 10; ++i)
    {
      struct timespec ts = { .tv_sec = 0, .tv_nsec = 100000000 };
      TEMP_FAILURE_RETRY (nanosleep (&ts, &ts));
    }

  seen = 1;
  pthread_setcancelstate (old, NULL);

  struct timespec ts = { .tv_sec = 0, .tv_nsec = 100000000 };
  TEMP_FAILURE_RETRY (nanosleep (&ts, &ts));

  exit (1);
}


static int
do_test (void)
{
  if (pthread_barrier_init (&b, NULL, 2) != 0)
   {
     puts ("barrier init failed");
     return 1;
   }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("thread creation failed");
      return 1;
    }

  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  if (pthread_cancel (th) != 0)
    {
      puts ("cancel failed");
      return 1;
    }

  void *status;
  if (pthread_join (th, &status) != 0)
    {
      puts ("join failed");
      return 1;
    }
  if (status != PTHREAD_CANCELED)
    {
      puts ("thread not canceled");
      return 1;
    }

  if (pthread_barrier_destroy (&b) != 0)
    {
      puts ("barrier_destroy failed");
      return 1;
    }

  if (seen != 1)
    {
      puts ("thread cancelled when PTHREAD_CANCEL_DISABLED");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
