/* Test program for timedout read/write lock functions.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xthread.h>


#define NWRITERS 15
#define WRITETRIES 10
#define NREADERS 15
#define READTRIES 15

static const struct timespec timeout = { 0,1000000 };
static const struct timespec delay = { 0, 1000000 };

#ifndef KIND
# define KIND PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP
#endif

/* A bogus clock value that tells the tests to use pthread_rwlock_timedrdlock
   and pthread_rwlock_timedwrlock rather than pthread_rwlock_clockrdlock and
   pthread_rwlock_clockwrlock.  */
#define CLOCK_USE_TIMEDLOCK (-1)

static pthread_rwlock_t lock;

struct thread_args
{
  int nr;
  clockid_t clockid;
  const char *fnname;
};

static void *
writer_thread (void *arg)
{
  struct thread_args *args = arg;
  const int nr = args->nr;
  const clockid_t clockid = args->clockid;
  const clockid_t clockid_for_get =
    (clockid == CLOCK_USE_TIMEDLOCK) ? CLOCK_REALTIME : clockid;
  const char *fnname = args->fnname;

  struct timespec ts;
  int n;

  for (n = 0; n < WRITETRIES; ++n)
    {
      int e;
      do
	{
	  xclock_gettime (clockid_for_get, &ts);

          ts = timespec_add (ts, timeout);
          ts = timespec_add (ts, timeout);

	  printf ("writer thread %d tries again\n", nr);

	  e = (clockid == CLOCK_USE_TIMEDLOCK)
	    ? pthread_rwlock_timedwrlock (&lock, &ts)
	    : pthread_rwlock_clockwrlock (&lock, clockid, &ts);
	  if (e != 0 && e != ETIMEDOUT)
            FAIL_EXIT1 ("%swrlock failed", fnname);
	}
      while (e == ETIMEDOUT);

      printf ("writer thread %d succeeded\n", nr);

      nanosleep (&delay, NULL);

      if (pthread_rwlock_unlock (&lock) != 0)
        FAIL_EXIT1 ("unlock for writer failed");

      printf ("writer thread %d released\n", nr);
    }

  return NULL;
}


static void *
reader_thread (void *arg)
{
  struct thread_args *args = arg;
  const int nr = args->nr;
  const clockid_t clockid = args->clockid;
  const clockid_t clockid_for_get =
    (clockid == CLOCK_USE_TIMEDLOCK) ? CLOCK_REALTIME : clockid;
  const char *fnname = args->fnname;

  struct timespec ts;
  int n;

  for (n = 0; n < READTRIES; ++n)
    {
      int e;
      do
	{
	  xclock_gettime (clockid_for_get, &ts);

          ts = timespec_add (ts, timeout);

	  printf ("reader thread %d tries again\n", nr);

	  if (clockid == CLOCK_USE_TIMEDLOCK)
	    e = pthread_rwlock_timedrdlock (&lock, &ts);
          else
	    e = pthread_rwlock_clockrdlock (&lock, clockid, &ts);
	  if (e != 0 && e != ETIMEDOUT)
            FAIL_EXIT1 ("%srdlock failed", fnname);
	}
      while (e == ETIMEDOUT);

      printf ("reader thread %d succeeded\n", nr);

      nanosleep (&delay, NULL);

      if (pthread_rwlock_unlock (&lock) != 0)
        FAIL_EXIT1 ("unlock for reader failed");

      printf ("reader thread %d released\n", nr);
    }

  return NULL;
}


static int
do_test_clock (clockid_t clockid, const char *fnname)
{
  pthread_t thwr[NWRITERS];
  pthread_t thrd[NREADERS];
  int n;
  pthread_rwlockattr_t a;

  if (pthread_rwlockattr_init (&a) != 0)
    FAIL_EXIT1 ("rwlockattr_t failed");

  if (pthread_rwlockattr_setkind_np (&a, KIND) != 0)
    FAIL_EXIT1 ("rwlockattr_setkind failed");

  if (pthread_rwlock_init (&lock, &a) != 0)
    FAIL_EXIT1 ("rwlock_init failed");

  /* Make standard error the same as standard output.  */
  dup2 (1, 2);

  /* Make sure we see all message, even those on stdout.  */
  setvbuf (stdout, NULL, _IONBF, 0);

  struct thread_args wargs[NWRITERS];
  for (n = 0; n < NWRITERS; ++n) {
    wargs[n].nr = n;
    wargs[n].clockid = clockid;
    wargs[n].fnname = fnname;
    thwr[n] = xpthread_create (NULL, writer_thread, &wargs[n]);
  }

  struct thread_args rargs[NREADERS];
  for (n = 0; n < NREADERS; ++n) {
    rargs[n].nr = n;
    rargs[n].clockid = clockid;
    rargs[n].fnname = fnname;
    thrd[n] = xpthread_create (NULL, reader_thread, &rargs[n]);
  }

  /* Wait for all the threads.  */
  for (n = 0; n < NWRITERS; ++n)
    xpthread_join (thwr[n]);
  for (n = 0; n < NREADERS; ++n)
    xpthread_join (thrd[n]);

  return 0;
}

static int
do_test (void)
{
  do_test_clock (CLOCK_USE_TIMEDLOCK, "timed");
  do_test_clock (CLOCK_REALTIME, "clock(realtime)");
  do_test_clock (CLOCK_MONOTONIC, "clock(monotonic)");

  return 0;
}

#define TIMEOUT 30
#include <support/test-driver.c>
