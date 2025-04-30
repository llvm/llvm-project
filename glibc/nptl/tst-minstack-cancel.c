/* Test cancellation with a minimal stack size.
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

/* Note: This test is similar to tst-minstack-exit, but is separate to
   avoid spurious test passes due to warm-up effects.  */

#include <limits.h>
#include <unistd.h>
#include <support/check.h>
#include <support/xthread.h>

static void *
threadfunc (void *closure)
{
  while (1)
    pause ();
  return NULL;
}

static int
do_test (void)
{
  pthread_attr_t attr;
  xpthread_attr_init (&attr);
  xpthread_attr_setstacksize (&attr, PTHREAD_STACK_MIN);
  pthread_t thr = xpthread_create (&attr, threadfunc, NULL);
  xpthread_cancel (thr);
  TEST_VERIFY (xpthread_join (thr) == PTHREAD_CANCELED);
  xpthread_attr_destroy (&attr);
  return 0;
}

#include <support/test-driver.c>
