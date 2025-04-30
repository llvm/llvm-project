/* C11 threads condition timed wait variable tests.
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

#include <threads.h>
#include <stdio.h>
#include <unistd.h>

#include <support/check.h>

/* Shared condition variable between child and parent.  */
static cnd_t cond;

/* Mutex needed to signal and wait threads.  */
static mtx_t mutex;

static int
signal_parent (void *arg)
{
  /* Acquire the lock so that cnd_signal does not run until
     cnd_timedwait has been called.  */
  if (mtx_lock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_lock failed");
  if (cnd_signal (&cond) != thrd_success)
    FAIL_EXIT1 ("cnd_signal failed");
  if (mtx_unlock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_unlock");

  thrd_exit (thrd_success);
}

static int
do_test (void)
{
  thrd_t id;
  struct timespec w_time;

  if (cnd_init (&cond) != thrd_success)
    FAIL_EXIT1 ("cnd_init failed");
  if (mtx_init (&mutex, mtx_plain) != thrd_success)
    FAIL_EXIT1 ("mtx_init failed");
  if (mtx_lock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_lock failed");

  if (clock_gettime (CLOCK_REALTIME, &w_time) != 0)
    FAIL_EXIT1 ("clock_gettime failed");

  /* This needs to be sufficiently long to prevent the cnd_timedwait
     call from timing out.  */
  w_time.tv_sec += 3600;

  if (thrd_create (&id, signal_parent, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd_create failed");

  if (cnd_timedwait (&cond, &mutex, &w_time) != thrd_success)
    FAIL_EXIT1 ("cnd_timedwait failed");

  if (thrd_join (id, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd_join failed");

  if (mtx_unlock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_unlock");

  mtx_destroy (&mutex);
  cnd_destroy (&cond);

  return 0;
}

#include <support/test-driver.c>
