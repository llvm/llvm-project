/* C11 threads recursive mutex tests.
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

static int
do_test (void)
{
  static mtx_t mutex;

  if (mtx_init (&mutex, mtx_plain | mtx_recursive) != thrd_success)
    FAIL_EXIT1 ("mtx_init failed");

  if (mtx_lock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_lock failed");

  /* Lock mutex second time, if not recursive should deadlock.  */
  if (mtx_lock (&mutex) != thrd_success)
    FAIL_EXIT1 ("mtx_lock failed");

  mtx_destroy (&mutex);

  return 0;
}

#include <support/test-driver.c>
