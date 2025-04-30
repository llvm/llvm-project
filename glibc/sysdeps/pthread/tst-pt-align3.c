/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2005.

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

static bool ok = true;
static pthread_once_t once = PTHREAD_ONCE_INIT;

static void
once_test (void)
{
  puts ("in once_test");

  if (TEST_STACK_ALIGN ())
    ok = false;
}

static int
do_test (void)
{
  puts ("in main");

  if (TEST_STACK_ALIGN ())
    ok = false;

  if (pthread_once (&once, once_test))
    {
      puts ("pthread once failed");
      return 1;
    }

  return ok ? 0 : 1;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
