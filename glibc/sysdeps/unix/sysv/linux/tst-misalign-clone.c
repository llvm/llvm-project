/* Verify that the clone wrapper properly aligns the child stack.
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
#include <stackinfo.h>
#include <support/xunistd.h>
#include <support/check.h>

static int
check_stack_alignment (void *arg)
{
  bool ok = true;

  puts ("in f");

  if (TEST_STACK_ALIGN ())
    ok = false;

  return ok ? 0 : 1;
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

#ifdef __ia64__
  extern int __clone2 (int (*__fn) (void *__arg), void *__child_stack_base,
		       size_t __child_stack_size, int __flags,
		       void *__arg, ...);
  pid_t p = __clone2 (check_stack_alignment, stack, STACK_SIZE, 0, 0);
#else
# if _STACK_GROWS_DOWN
  pid_t p = clone (check_stack_alignment, stack + STACK_SIZE, 0, 0);
# elif _STACK_GROWS_UP
  pid_t p = clone (check_stack_alignment, stack, 0, 0);
# else
#  error "Define either _STACK_GROWS_DOWN or _STACK_GROWS_UP"
# endif
#endif

  /* Clone must not fail.  */
  TEST_VERIFY_EXIT (p != -1);

  int e;
  xwaitpid (p, &e, __WCLONE);
  if (!WIFEXITED (e))
    {
      if (WIFSIGNALED (e))
	printf ("died from signal %s\n", strsignal (WTERMSIG (e)));
     FAIL_EXIT1 ("process did not terminate correctly");
    }

  if (WEXITSTATUS (e) != 0)
    FAIL_EXIT1 ("exit code %d", WEXITSTATUS (e));

  return 0;
}

#include <support/test-driver.c>
