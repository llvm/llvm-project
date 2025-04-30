/* Test malloc with concurrent thread termination.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* This thread spawns a number of outer threads, equal to the arena
   limit.  The outer threads run a loop which start and join two
   different kinds of threads: the first kind allocates (attaching an
   arena to the thread; malloc_first_thread) and waits, the second
   kind waits and allocates (wait_first_threads).  Both kinds of
   threads exit immediately after waiting.  The hope is that this will
   exhibit races in thread termination and arena management,
   particularly related to the arena free list.  */

#include <errno.h>
#include <malloc.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <support/support.h>
#include <support/xthread.h>
#include <support/test-driver.h>

static bool termination_requested;
static int inner_thread_count = 4;
static size_t malloc_size = 32;

static void
__attribute__ ((noinline, noclone))
unoptimized_free (void *ptr)
{
  free (ptr);
}

static void *
malloc_first_thread (void * closure)
{
  pthread_barrier_t *barrier = closure;
  void *ptr = xmalloc (malloc_size);
  xpthread_barrier_wait (barrier);
  unoptimized_free (ptr);
  return NULL;
}

static void *
wait_first_thread (void * closure)
{
  pthread_barrier_t *barrier = closure;
  xpthread_barrier_wait (barrier);
  void *ptr = xmalloc (malloc_size);
  unoptimized_free (ptr);
  return NULL;
}

static void *
outer_thread (void *closure)
{
  pthread_t *threads = xcalloc (sizeof (*threads), inner_thread_count);
  while (!__atomic_load_n (&termination_requested, __ATOMIC_RELAXED))
    {
      pthread_barrier_t barrier;
      xpthread_barrier_init (&barrier, NULL, inner_thread_count + 1);
      for (int i = 0; i < inner_thread_count; ++i)
        {
          void *(*func) (void *);
          if ((i  % 2) == 0)
            func = malloc_first_thread;
          else
            func = wait_first_thread;
          threads[i] = xpthread_create (NULL, func, &barrier);
        }
      xpthread_barrier_wait (&barrier);
      for (int i = 0; i < inner_thread_count; ++i)
        xpthread_join (threads[i]);
      xpthread_barrier_destroy (&barrier);
    }

  free (threads);

  return NULL;
}

static int
do_test (void)
{
  /* The number of threads should be smaller than the number of
     arenas, so that there will be some free arenas to add to the
     arena free list.  */
  enum { outer_thread_count = 2 };
  if (mallopt (M_ARENA_MAX, 8) == 0)
    {
      printf ("error: mallopt (M_ARENA_MAX) failed\n");
      return 1;
    }

  /* Leave some room for shutting down all threads gracefully.  */
  int timeout = 3;
  if (timeout > DEFAULT_TIMEOUT)
    timeout = DEFAULT_TIMEOUT - 1;

  pthread_t *threads = xcalloc (sizeof (*threads), outer_thread_count);
  for (long i = 0; i < outer_thread_count; ++i)
    threads[i] = xpthread_create (NULL, outer_thread, NULL);

  struct timespec ts = {timeout, 0};
  if (nanosleep (&ts, NULL))
    {
      printf ("error: error: nanosleep: %m\n");
      abort ();
    }

  __atomic_store_n (&termination_requested, true, __ATOMIC_RELAXED);

  for (long i = 0; i < outer_thread_count; ++i)
    xpthread_join (threads[i]);
  free (threads);

  return 0;
}

#include <support/test-driver.c>
