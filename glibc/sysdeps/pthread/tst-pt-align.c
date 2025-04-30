/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <tst-stack-align.h>

static void *
tf (void *arg)
{
  bool ok = true;

  puts ("in thread");

  if (TEST_STACK_ALIGN ())
    ok = false;

  return ok ? NULL : (void *) -1l;
}

static int
do_test (void)
{
  bool ok = true;

  puts ("in main");

  if (TEST_STACK_ALIGN ())
    ok = false;

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  void *res;
  if (pthread_join (th, &res) != 0)
    {
      puts ("join failed");
      return 1;
    }

  if (res != NULL)
    ok = false;

  return ok ? 0 : 1;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
