/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define N 100

static pthread_once_t once = PTHREAD_ONCE_INIT;

static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;

static pthread_barrier_t bar;

static int global;
static int cl_called;

static void
once_handler1 (void)
{
  if (pthread_mutex_lock (&mut) != 0)
    {
      puts ("once_handler1: mutex_lock failed");
      exit (1);
    }
  puts ("once_handler1: locked");

  int r = pthread_barrier_wait (&bar);
  if (r != 0 && r!= PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("once_handler1: barrier_wait failed");
      exit (1);
    }

  puts ("once_handler1: going to wait on cond");

  pthread_cond_wait (&cond, &mut);

  /* We should never get here.  */
  exit (42);
}

static void
once_handler2 (void)
{
  global = 1;
}


static void
cl (void *arg)
{
  cl_called = 1;
}


static void *
tf (void *arg)
{
  pthread_cleanup_push (cl, NULL)

  pthread_once (&once, once_handler1);

  pthread_cleanup_pop (0);

  /* We should never get here.  */
  puts ("pthread_once in tf returned");
  exit (1);
}


static int
do_test (void)
{
  pthread_t th;

  if (pthread_barrier_init (&bar, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("first create failed");
      return 1;
    }

  int r = pthread_barrier_wait (&bar);
  if (r != 0 && r!= PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  if (pthread_mutex_lock (&mut) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }
  /* We unlock the mutex so that we catch the case where the pthread_cond_wait
     call incorrectly resumes and tries to get the mutex.  */
  if (pthread_mutex_unlock (&mut) != 0)
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  /* Cancel the thread.  */
  puts ("going to cancel");
  if (pthread_cancel (th) != 0)
    {
      puts ("cancel failed");
      return 1;
    }

  void *result;
  pthread_join (th, &result);
  if (result != PTHREAD_CANCELED)
    {
      puts ("join didn't return PTHREAD_CANCELED");
      return 1;
    }
  puts ("joined successfully");

  printf ("once = %d\n", *(int *) &once);

  if (cl_called != 1)
    {
      puts ("cleanup handler not called");
      return 1;
    }

  pthread_once (&once, once_handler2);

  if (global != 1)
    {
      puts ("global still 0");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
