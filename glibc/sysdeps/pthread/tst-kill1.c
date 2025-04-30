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

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>


static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_barrier_t b;

static void *
tf (void *a)
{
  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("child: mutex_lock failed");
      exit (1);
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("child: barrier_wait failed");
      exit (1);
    }

  /* This call should never return.  */
  pthread_cond_wait (&c, &m);

  return NULL;
}


int
do_test (void)
{
  pthread_t th;

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      exit (1);
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("mutex_lock failed");
      exit (1);
    }

  /* Send the thread a signal which it doesn't catch and which will
     cause the process to terminate.  */
  if (pthread_kill (th, SIGUSR1) != 0)
    {
      puts ("kill failed");
      exit (1);
    }

  /* This call should never return.  */
  pthread_join (th, NULL);

  return 0;
}


#define EXPECTED_SIGNAL SIGUSR1
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
