/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <stackinfo.h>

static int
f (void *arg)
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
  bool ok = true;

  puts ("in main");

  if (TEST_STACK_ALIGN ())
    ok = false;

#ifdef __ia64__
  extern int __clone2 (int (*__fn) (void *__arg), void *__child_stack_base,
		       size_t __child_stack_size, int __flags,
		       void *__arg, ...);
  char st[256 * 1024];
  pid_t p = __clone2 (f, st, sizeof (st), 0, 0);
#else
  char st[128 * 1024] __attribute__ ((aligned));
# if _STACK_GROWS_DOWN
  pid_t p = clone (f, st + sizeof (st), 0, 0);
# elif _STACK_GROWS_UP
  pid_t p = clone (f, st, 0, 0);
# else
#  error "Define either _STACK_GROWS_DOWN or _STACK_GROWS_UP"
# endif
#endif
  if (p == -1)
    {
      printf("clone failed: %m\n");
      return 1;
    }

  int e;
  if (waitpid (p, &e, __WCLONE) != p)
    {
      puts ("waitpid failed");
      kill (p, SIGKILL);
      return 1;
    }
  if (!WIFEXITED (e))
    {
      if (WIFSIGNALED (e))
	printf ("died from signal %s\n", strsignal (WTERMSIG (e)));
      else
	puts ("did not terminate correctly");
      return 1;
    }
  if (WEXITSTATUS (e) != 0)
    ok = false;

  return ok ? 0 : 1;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
