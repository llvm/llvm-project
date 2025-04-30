/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* This tests that a writer that is preferred -- but times out due to a
   reader being present -- does not miss to wake other readers blocked on the
   writer's pending lock acquisition.  */

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* The bug existed in the code that strictly prefers writers over readers.  */
static pthread_rwlock_t r = PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP;

static void *
writer (void *arg)
{
  struct timespec ts;
  if (clock_gettime (CLOCK_REALTIME, &ts) != 0)
    {
      puts ("clock_gettime failed");
      exit (EXIT_FAILURE);
    }
  ts.tv_sec += 1;
  int e = pthread_rwlock_timedwrlock (&r, &ts);
  if (e != ETIMEDOUT)
    {
      puts ("timedwrlock did not time out");
      exit (EXIT_FAILURE);
    }
  return NULL;
}

static void *
reader (void *arg)
{
  /* This isn't a reliable way to get the interleaving we need (because a
     failed trylock doesn't synchronize with the writer, and because we could
     try to lock after the writer has already timed out).  However, both will
     just lead to false positives.  */
  int e;
  while ((e = pthread_rwlock_tryrdlock (&r)) != EBUSY)
    {
      if (e != 0)
	exit (EXIT_FAILURE);
      pthread_rwlock_unlock (&r);
    }
  e = pthread_rwlock_rdlock (&r);
  if (e != 0)
    {
      puts ("reader rdlock failed");
      exit (EXIT_FAILURE);
    }
  pthread_rwlock_unlock (&r);
  return NULL;
}


static int
do_test (void)
{
  /* Grab a rdlock, then create a writer and a reader, and wait until they
     finished.  */

  if (pthread_rwlock_rdlock (&r) != 0)
    {
      puts ("initial rdlock failed");
      return 1;
    }

  pthread_t thw;
  if (pthread_create (&thw, NULL, writer, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }
  pthread_t thr;
  if (pthread_create (&thr, NULL, reader, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  if (pthread_join (thw, NULL) != 0)
    {
      puts ("writer join failed");
      return 1;
    }
  if (pthread_join (thr, NULL) != 0)
    {
      puts ("reader join failed");
      return 1;
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
