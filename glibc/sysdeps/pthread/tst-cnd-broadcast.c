/* C11 threads condition broadcast variable tests.
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
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include <support/check.h>

/* Condition variable where child threads will wait.  */
static cnd_t cond;

/* Mutex to control wait on cond.  */
static mtx_t mutex;

/* Number of threads which have entered the cnd_wait region.  */
static unsigned int waiting_threads;

/* Code executed by each thread.  */
static int
child_wait (void* data)
{
  /* Wait until parent thread sends broadcast here.  */
  mtx_lock (&mutex);
  ++waiting_threads;
  cnd_wait (&cond, &mutex);
  mtx_unlock (&mutex);

  thrd_exit (thrd_success);
}

#define N 5

static int
do_test (void)
{
  thrd_t ids[N];
  unsigned char i;

  if (cnd_init (&cond) != thrd_success)
    FAIL_EXIT1 ("cnd_init failed");
  if (mtx_init (&mutex, mtx_plain) != thrd_success)
    FAIL_EXIT1 ("mtx_init failed");

  /* Create N new threads.  */
  for (i = 0; i < N; ++i)
    {
      if (thrd_create (&ids[i], child_wait, NULL) != thrd_success)
	FAIL_EXIT1 ("thrd_create failed");
    }

  /* Wait for other threads to reach their wait func.  */
  while (true)
    {
      mtx_lock (&mutex);
      TEST_VERIFY (waiting_threads <= N);
      bool done_waiting = waiting_threads == N;
      mtx_unlock (&mutex);
      if (done_waiting)
	break;
      thrd_sleep (&((struct timespec){.tv_nsec = 100 * 1000 * 1000}), NULL);
    }

  mtx_lock (&mutex);
  if (cnd_broadcast (&cond) != thrd_success)
    FAIL_EXIT1 ("cnd_broadcast failed");
  mtx_unlock (&mutex);

  for (i = 0; i < N; ++i)
    {
      if (thrd_join (ids[i], NULL) != thrd_success)
	FAIL_EXIT1 ("thrd_join failed");
    }

  mtx_destroy (&mutex);
  cnd_destroy (&cond);

  return 0;
}

#include <support/test-driver.c>
