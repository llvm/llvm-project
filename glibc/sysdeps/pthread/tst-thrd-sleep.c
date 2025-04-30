/* C11 threads thread sleep tests.
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
sleep_thrd (void *arg)
{
  struct timespec const *tl = (struct timespec const *) arg;
  if (thrd_sleep (tl, NULL) != 0)
    FAIL_EXIT1 ("thrd_sleep failed");

  thrd_exit (thrd_success);
}

static int
do_test (void)
{
  thrd_t id;
  struct timespec wait_time = {.tv_sec = 3};

  if (thrd_create (&id, sleep_thrd, (void *) (&wait_time)) != thrd_success)
    FAIL_EXIT1 ("thrd_create failed");

  if (thrd_join (id, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd failed");

  return 0;
}

#include <support/test-driver.c>
