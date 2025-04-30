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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <support/check.h>
#include <support/timespec.h>
#include <support/xunistd.h>

#ifdef ENABLE_PP
#include "tst-tpp.h"
#endif


/* A bogus clock value that tells run_test to use pthread_mutex_timedlock
   rather than pthread_mutex_clocklock.  */
#define CLOCK_USE_TIMEDLOCK (-1)

static void
do_test_clock (clockid_t clockid)
{
  const clockid_t clockid_for_get =
    (clockid == CLOCK_USE_TIMEDLOCK) ? CLOCK_REALTIME : clockid;
  size_t ps = sysconf (_SC_PAGESIZE);
  char tmpfname[] = "/tmp/tst-mutex9.XXXXXX";
  char data[ps];
  void *mem;
  int fd;
  pthread_mutex_t *m;
  pthread_mutexattr_t a;
  pid_t pid;

  fd = mkstemp (tmpfname);
  if (fd == -1)
      FAIL_EXIT1 ("cannot open temporary file: %m\n");

  /* Make sure it is always removed.  */
  unlink (tmpfname);

  /* Create one page of data.  */
  memset (data, '\0', ps);

  /* Write the data to the file.  */
  xwrite (fd, data, ps);

  mem = xmmap (NULL, ps, PROT_READ | PROT_WRITE, MAP_SHARED, fd);

  m = (pthread_mutex_t *) (((uintptr_t) mem + __alignof (pthread_mutex_t))
			   & ~(__alignof (pthread_mutex_t) - 1));

  TEST_COMPARE (pthread_mutexattr_init (&a), 0);

  TEST_COMPARE (pthread_mutexattr_setpshared (&a, PTHREAD_PROCESS_SHARED), 0);

  TEST_COMPARE (pthread_mutexattr_settype (&a, PTHREAD_MUTEX_RECURSIVE), 0);

#if defined ENABLE_PI
  TEST_COMPARE (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_INHERIT), 0);
#elif defined ENABLE_PP
  TEST_COMPARE (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_PROTECT), 0);
  TEST_COMPARE (pthread_mutexattr_setprioceiling (&a, 6), 0);
#endif

  int e;
  if ((e = pthread_mutex_init (m, &a)) != 0)
    {
#ifdef ENABLE_PI
      if (e == ENOTSUP)
        FAIL_UNSUPPORTED ("PI mutexes unsupported");
#endif
      FAIL_EXIT1 ("mutex_init failed");
    }

  TEST_COMPARE (pthread_mutex_lock (m), 0);

  TEST_COMPARE (pthread_mutexattr_destroy (&a), 0);

  puts ("going to fork now");
  pid = xfork ();
  if (pid == 0)
    {
      if (pthread_mutex_trylock (m) == 0)
        FAIL_EXIT1 ("child: mutex_trylock succeeded");

      if (pthread_mutex_unlock (m) == 0)
        FAIL_EXIT1 ("child: mutex_unlock succeeded");

      const struct timespec ts = timespec_add (xclock_now (clockid_for_get),
                                               make_timespec (0, 500000000));

      if (clockid == CLOCK_USE_TIMEDLOCK)
        TEST_COMPARE (pthread_mutex_timedlock (m, &ts), ETIMEDOUT);
      else
        TEST_COMPARE (pthread_mutex_clocklock (m, clockid, &ts), ETIMEDOUT);

      alarm (1);

      pthread_mutex_lock (m);

      puts ("child: mutex_lock returned");

      exit (0);
    }

  sleep (2);

  int status;
  if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
    FAIL_EXIT1 ("waitpid failed");
  if (! WIFSIGNALED (status))
    FAIL_EXIT1 ("child not killed by signal");
  TEST_COMPARE (WTERMSIG (status), SIGALRM);
}

static int
do_test (void)
{
#ifdef ENABLE_PP
  init_tpp_test ();
#endif

  do_test_clock (CLOCK_USE_TIMEDLOCK);
  do_test_clock (CLOCK_REALTIME);
#ifndef ENABLE_PI
  do_test_clock (CLOCK_MONOTONIC);
#endif
  return 0;
}

#include <support/test-driver.c>
