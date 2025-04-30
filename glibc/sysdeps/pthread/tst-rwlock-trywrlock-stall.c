/* Bug 23844: Test for pthread_rwlock_trywrlock stalls.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* For a full analysis see comments in tst-rwlock-tryrdlock-stall.c.

   Summary for the pthread_rwlock_trywrlock() stall:

   The stall is caused by pthread_rwlock_trywrlock setting
   __wrphase_futex futex to 1 and loosing the
   PTHREAD_RWLOCK_FUTEX_USED bit.

   The fix for bug 23844 ensures that waiters on __wrphase_futex are
   correctly woken.  Before the fix the test stalls as readers can
   wait forever on  __wrphase_futex.  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <support/xthread.h>
#include <errno.h>

/* We need only one lock to reproduce the issue. We will need multiple
   threads to get the exact case where we have a read, try, and unlock
   all interleaving to produce the case where the readers are waiting
   and the try clears the PTHREAD_RWLOCK_FUTEX_USED bit and a
   subsequent unlock fails to wake them.  */
pthread_rwlock_t onelock;

/* The number of threads is arbitrary but empirically chosen to have
   enough threads that we see the condition where waiting readers are
   not woken by a successful unlock.  */
#define NTHREADS 32

_Atomic int do_exit;

void *
run_loop (void *arg)
{
  int i = 0, ret;
  while (!do_exit)
    {
      /* Arbitrarily choose if we are the writer or reader.  Choose a
	 high enough ratio of readers to writers to make it likely
	 that readers block (and eventually are susceptable to
	 stalling).

         If we are a writer, take the write lock, and then unlock.
	 If we are a reader, try the lock, then lock, then unlock.  */
      if ((i % 8) != 0)
	{
	  if ((ret = pthread_rwlock_trywrlock (&onelock)) != 0)
	    {
	      if (ret == EBUSY)
		xpthread_rwlock_wrlock (&onelock);
	      else
		exit (EXIT_FAILURE);
	    }
	}
      else
	xpthread_rwlock_rdlock (&onelock);
      /* Thread does some work and then unlocks.  */
      xpthread_rwlock_unlock (&onelock);
      i++;
    }
  return NULL;
}

int
do_test (void)
{
  int i;
  pthread_t tids[NTHREADS];
  xpthread_rwlock_init (&onelock, NULL);
  for (i = 0; i < NTHREADS; i++)
    tids[i] = xpthread_create (NULL, run_loop, NULL);
  /* Run for some amount of time.  The pthread_rwlock_tryrwlock stall
     is very easy to trigger and happens in seconds under the test
     conditions.  */
  sleep (10);
  /* Then exit.  */
  printf ("INFO: Exiting...\n");
  do_exit = 1;
  /* If any readers stalled then we will timeout waiting for them.  */
  for (i = 0; i < NTHREADS; i++)
    xpthread_join (tids[i]);
  printf ("INFO: Done.\n");
  xpthread_rwlock_destroy (&onelock);
  printf ("PASS: No pthread_rwlock_tryrwlock stalls detected.\n");
  return 0;
}

#include <support/test-driver.c>
