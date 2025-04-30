/* C11 threads thread detach tests.
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
#include <time.h>
#include <stdio.h>
#include <unistd.h>

#include <support/check.h>

static int
detach_thrd (void *arg)
{
  if (thrd_detach (thrd_current ()) != thrd_success)
    FAIL_EXIT1 ("thrd_detach failed");
  thrd_exit (thrd_success);
}

static int
do_test (void)
{
  thrd_t id;

  /* Create new thread.  */
  if (thrd_create (&id, detach_thrd, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd_create failed");

  /* Give some time so the thread can finish.  */
  thrd_sleep (&(struct timespec) {.tv_sec = 2}, NULL);

  if (thrd_join (id, NULL) == thrd_success)
    FAIL_EXIT1 ("thrd_join succeed where it should fail");

  return 0;
}

#include <support/test-driver.c>
