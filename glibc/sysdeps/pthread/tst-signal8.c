/* Tests for sigisemptyset and pthread_sigmask.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <signal.h>

#include <support/check.h>
#include <support/xthread.h>

static void *
tf (void *arg)
{
  {
    sigset_t set;
    sigemptyset (&set);
    TEST_COMPARE (pthread_sigmask (SIG_BLOCK, 0, &set), 0);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  {
    sigset_t set;
    sigfillset (&set);
    TEST_COMPARE (pthread_sigmask (SIG_BLOCK, 0, &set), 0);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  return NULL;
}

static int
do_test (void)
{
  /* Ensure current SIG_BLOCK mask empty.  */
  {
    sigset_t set;
    sigemptyset (&set);
    TEST_COMPARE (sigprocmask (SIG_BLOCK, &set, 0), 0);
  }

  {
    pthread_t thr = xpthread_create (NULL, tf, NULL);
    xpthread_join (thr);
  }

  return 0;
}

#include <support/test-driver.c>
