/* Verify that the clone child stack is properly aligned.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <tst-stack-align.h>
#include <clone_internal.h>
#include <support/xunistd.h>
#include <support/check.h>

static int
f (void *arg)
{
  puts ("in f");

  return TEST_STACK_ALIGN () ? 1 : 0;
}

static int
do_test (void)
{
  puts ("in main");

  if (TEST_STACK_ALIGN ())
    FAIL_EXIT1 ("stack alignment failed");

#ifdef __ia64__
# define STACK_SIZE 256 * 1024
#else
# define STACK_SIZE 128 * 1024
#endif
  char st[STACK_SIZE] __attribute__ ((aligned));
  struct clone_args clone_args =
    {
      .stack = (uintptr_t) st,
      .stack_size = sizeof (st),
    };
  pid_t p = __clone_internal (&clone_args, f, 0);
  TEST_VERIFY (p != -1);

  int e;
  xwaitpid (p, &e, __WCLONE);
  TEST_VERIFY (WIFEXITED (e));
  TEST_COMPARE (WEXITSTATUS (e), 0);
  return 0;
}

#include <support/test-driver.c>
