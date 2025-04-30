/* Verify that __clone_internal properly aligns the child stack.
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
#include <libc-pointer-arith.h>
#include <tst-stack-align.h>
#include <clone_internal.h>
#include <support/xunistd.h>
#include <support/check.h>

static int
check_stack_alignment (void *arg)
{
  puts ("in f");

  return TEST_STACK_ALIGN () ? 1 : 0;
}

static int
do_test (void)
{
  puts ("in do_test");

  if (TEST_STACK_ALIGN ())
    FAIL_EXIT1 ("stack isn't aligned\n");

#ifdef __ia64__
# define STACK_SIZE (256 * 1024)
#else
# define STACK_SIZE (128 * 1024)
#endif
  char st[STACK_SIZE + 1];
  /* NB: Align child stack to 1 byte.  */
  char *stack = PTR_ALIGN_UP (&st[0], 2) + 1;
  struct clone_args clone_args =
    {
      .stack = (uintptr_t) stack,
      .stack_size = STACK_SIZE,
    };
  pid_t p = __clone_internal (&clone_args, check_stack_alignment, 0);

  /* Clone must not fail.  */
  TEST_VERIFY_EXIT (p != -1);

  int e;
  xwaitpid (p, &e, __WCLONE);
  TEST_VERIFY (WIFEXITED (e));
  TEST_COMPARE (WEXITSTATUS (e), 0);

  return 0;
}

#include <support/test-driver.c>
