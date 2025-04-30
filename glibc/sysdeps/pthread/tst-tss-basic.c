/* C11 threads specific storage tests.
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

/* Thread specific storage.  */
static tss_t key;

#define TSS_VALUE (void*) 0xFF

static int
tss_thrd (void *arg)
{
  if (tss_create (&key, NULL) != thrd_success)
    FAIL_EXIT1 ("tss_create failed");

  if (tss_set (key, TSS_VALUE) != thrd_success)
    FAIL_EXIT1 ("tss_set failed");

  void *value = tss_get (key);
  if (value == 0)
    FAIL_EXIT1 ("tss_get failed");
  if (value != TSS_VALUE)
    FAIL_EXIT1 ("tss_get returned %p, expected %p", value, TSS_VALUE);

  thrd_exit (thrd_success);
}

static int
do_test (void)
{
  /* Setting an invalid key should return an error.  */
  if (tss_set (key, TSS_VALUE) == thrd_success)
    FAIL_EXIT1 ("tss_set succeed where it should have failed");

  if (tss_create (&key, NULL) != thrd_success)
    FAIL_EXIT1 ("tss_create failed");

  thrd_t id;
  if (thrd_create (&id, tss_thrd, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd_create failed");

  if (thrd_join (id, NULL) != thrd_success)
    FAIL_EXIT1 ("thrd failed");

  /* The value set in tss_thrd should not be visible here.  */
  void *value = tss_get (key);
  if (value != 0)
    FAIL_EXIT1 ("tss_get succeed where it should have failed");

  tss_delete (key);

  return 0;
}

#include <support/test-driver.c>
