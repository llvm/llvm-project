/* C11 threads condition variable tests.
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
signal_parent (void)
{
  /* Acquire the lock so that cnd_signal does not run until
     cnd_timedwait has been called.  */
  if (mtx_lock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_lock failed");
  if (cnd_signal (&cond) != thrd_success)
    FAIL_EXIT1 ("cnd_signal");
  if (mtx_unlock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_unlock");

  thrd_exit (thrd_success);
}

static int
do_test (void)
{
  thrd_t id;

  if (cnd_init (&cond) != thrd_success)
    FAIL_EXIT1 ("cnd_init failed");
  if (mtx_init (&mutex, mtx_plain) != thrd_success)
    FAIL_EXIT1 ("mtx_init failed");

  if (mtx_lock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_lock failed");

  if (thrd_create (&id, (thrd_start_t) signal_parent, NULL)
      != thrd_success)
    FAIL_EXIT1 ("thrd_create failed");

  if (cnd_wait (&cond, &mutex) != thrd_success)
    FAIL_EXIT1 ("cnd_wait failed");

  /* Joining is not mandatory here, but still done to assure child thread
     ends correctly.  */
  if (thrd_join (id, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd_join failed");

  if (mtx_unlock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_unlock");

  mtx_destroy (&mutex);
  cnd_destroy (&cond);

  return 0;
}

#include <support/test-driver.c>
