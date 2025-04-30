/* C11 threads timed mutex tests.
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

/* Shared mutex between child and parent.  */
static mtx_t mutex;

/* Shared counter to check possible race conditions.  */
static char shrd_counter;

/* Maximum amount of time waiting for mutex.  */
static struct timespec wait_time;

/* Function to choose an action to do, depending on mtx_timedlock
   return value.  */
static inline void
choose_action (int action, char* thread_name)
{
  switch (action)
    {
      case thrd_success:
        ++shrd_counter;

	if (mtx_unlock (&mutex) != thrd_success)
	  FAIL_EXIT1 ("mtx_unlock failed");
      break;

      case thrd_timedout:
        break;

      case thrd_error:
	FAIL_EXIT1 ("%s lock error", thread_name);
        break;
    }
}

static int
child_add (void *arg)
{
  char child_name[] = "child";

  /* Try to lock mutex.  */
  choose_action (mtx_timedlock (&mutex, &wait_time), child_name);
  thrd_exit (thrd_success);
}

static int
do_test (void)
{
  thrd_t id;
  char parent_name[] = "parent";

  if (mtx_init (&mutex, mtx_timed) != thrd_success)
    FAIL_EXIT1 ("mtx_init failed");

  if (clock_gettime (CLOCK_REALTIME, &wait_time) != 0)
    FAIL_EXIT1 ("clock_gettime failed");
  /* Tiny amount of time, to assure that if any thread finds it busy.
     It will receive thrd_timedout.  */
  wait_time.tv_nsec += 1;
  if (wait_time.tv_nsec == 1000 * 1000 * 1000)
    {
      wait_time.tv_sec += 1;
      wait_time.tv_nsec = 0;
    }

  if (thrd_create (&id, child_add, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd_create failed");

  choose_action (mtx_timedlock (&mutex, &wait_time), parent_name);

  if (thrd_join (id, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd_join failed");

  if (shrd_counter != 2 && shrd_counter != 1)
    FAIL_EXIT1 ("shrd_counter != {1,2} (%d)", shrd_counter);

  mtx_destroy (&mutex);

  return 0;
}

#include <support/test-driver.c>
