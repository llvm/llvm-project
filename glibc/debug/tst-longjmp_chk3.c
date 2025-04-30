/* Make sure longjmp fortification catches bad signal stacks.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <setjmp.h>
#include <signal.h>
#include <string.h>

static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static char buf[SIGSTKSZ * 4];
static jmp_buf jb;

static void
handler (int sig)
{
  if (sig == SIGUSR1)
    {
      if (setjmp (jb) != 0)
	{
	  puts ("setjmp should not have been called");
	  kill (getpid (), SIGTERM);
	}
    }
  else if (sig == SIGABRT)
    {
      /* Yeah it worked.  */
      _exit (0);
    }
}

static int
do_test (void)
{
  stack_t ss;

  set_fortify_handler (handler);

  /* Create a valid signal stack and enable it.  */
  ss.ss_sp = buf;
  ss.ss_size = sizeof (buf);
  ss.ss_flags = 0;
  if (sigaltstack (&ss, NULL) < 0)
    {
      printf ("first sigaltstack failed: %m\n");
      return 1;
    }

  /* Trigger the signal handler which will create a jmpbuf that points to the
     end of the signal stack.  */
  signal (SIGUSR1, handler);
  kill (getpid (), SIGUSR1);

  /* Shrink the signal stack so the jmpbuf is now invalid.
     We adjust the start & end to handle stacks that grow up & down.  */
  ss.ss_sp = buf + sizeof (buf) / 2;
  ss.ss_size = sizeof (buf) / 4;
  if (sigaltstack (&ss, NULL) < 0)
    {
      printf ("second sigaltstack failed: %m\n");
      return 1;
    }

  /* This should fail.  */
  longjmp (jb, 1);

  puts ("longjmp returned and shouldn't");
  return 1;
}
