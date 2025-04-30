/* Test program for a read-phase / write-phase explicit hand-over.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <error.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <atomic.h>
#include <support/xthread.h>

/* We realy want to set threads to 2 to reproduce this issue. The goal
   is to have one primary writer and a single reader, and to hit the
   bug that happens in the interleaving of those two phase transitions.
   However, on most hardware, adding a second writer seems to help the
   interleaving happen slightly more often, say 20% of the time.  On a
   16 core ppc64 machine this fails 100% of the time with an unpatched
   glibc.  On a 8 core x86_64 machine this fails ~93% of the time, but
   it doesn't fail at all on a 4 core system, so having available
   unloaded cores makes a big difference in reproducibility.  On an 8
   core qemu/kvm guest the reproducer reliability drops to ~10%.  */
#define THREADS 3

#define KIND PTHREAD_RWLOCK_PREFER_READER_NP

static pthread_rwlock_t lock;
static int done = 0;

static void*
tf (void* arg)
{
  while (atomic_load_relaxed (&done) == 0)
    {
      int rcnt = 0;
      int wcnt = 100;
      if ((uintptr_t) arg == 0)
	{
	  rcnt = 1;
	  wcnt = 1;
	}

      do
	{
	  if (wcnt)
	    {
	      xpthread_rwlock_wrlock (&lock);
	      xpthread_rwlock_unlock (&lock);
	      wcnt--;
	  }
	  if (rcnt)
	    {
	      xpthread_rwlock_rdlock (&lock);
	      xpthread_rwlock_unlock (&lock);
	      rcnt--;
	  }
	}
      while ((atomic_load_relaxed (&done) == 0) && (rcnt + wcnt > 0));

    }
    return NULL;
}



static int
do_test (void)
{
  pthread_t thr[THREADS];
  int n;
  pthread_rwlockattr_t attr;

  xpthread_rwlockattr_init (&attr);
  xpthread_rwlockattr_setkind_np (&attr, KIND);

  xpthread_rwlock_init (&lock, &attr);

  /* Make standard error the same as standard output.  */
  dup2 (1, 2);

  /* Make sure we see all message, even those on stdout.  */
  setvbuf (stdout, NULL, _IONBF, 0);

  for (n = 0; n < THREADS; ++n)
    thr[n] = xpthread_create (NULL, tf, (void *) (uintptr_t) n);

  struct timespec delay;
  delay.tv_sec = 10;
  delay.tv_nsec = 0;
  nanosleep (&delay, NULL);
  atomic_store_relaxed (&done, 1);

  /* Wait for all the threads.  */
  for (n = 0; n < THREADS; ++n)
    xpthread_join (thr[n]);

  return 0;
}

#include <support/test-driver.c>
