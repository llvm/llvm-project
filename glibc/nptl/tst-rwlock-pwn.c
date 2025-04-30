/* Test rwlock with PREFER_WRITER_NONRECURSIVE_NP (bug 23861).
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <support/xthread.h>

/* We choose 10 iterations because this happens to be able to trigger the
   stall on contemporary hardware.  */
#define LOOPS 10
/* We need 3 threads to trigger bug 23861.  One thread as a writer, and
   two reader threads.  The test verifies that the second-to-last reader
   is able to notify the *last* reader that it should be done waiting.
   If the second-to-last reader fails to notify the last reader or does
   so incorrectly then the last reader may stall indefinitely.  */
#define NTHREADS 3

_Atomic int do_exit;
pthread_rwlockattr_t mylock_attr;
pthread_rwlock_t mylock;

void *
run_loop (void *a)
{
  while (!do_exit)
    {
      if (random () & 1)
	{
	  xpthread_rwlock_wrlock (&mylock);
	  xpthread_rwlock_unlock (&mylock);
	}
      else
	{
	  xpthread_rwlock_rdlock (&mylock);
	  xpthread_rwlock_unlock (&mylock);
	}
    }
  return NULL;
}

int
do_test (void)
{
  xpthread_rwlockattr_init (&mylock_attr);
  xpthread_rwlockattr_setkind_np (&mylock_attr,
				  PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
  xpthread_rwlock_init (&mylock, &mylock_attr);

  for (int n = 0; n < LOOPS; n++)
    {
      pthread_t tids[NTHREADS];
      do_exit = 0;
      for (int i = 0; i < NTHREADS; i++)
	tids[i] = xpthread_create (NULL, run_loop, NULL);
      /* Let the threads run for some time.  */
      sleep (1);
      printf ("Exiting...");
      fflush (stdout);
      do_exit = 1;
      for (int i = 0; i < NTHREADS; i++)
	xpthread_join (tids[i]);
      printf ("done.\n");
    }
  pthread_rwlock_destroy (&mylock);
  pthread_rwlockattr_destroy (&mylock_attr);
  return 0;
}

#define TIMEOUT (DEFAULT_TIMEOUT + 3 * LOOPS)
#include <support/test-driver.c>
