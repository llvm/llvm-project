/* Test backtrace and backtrace_symbols for signal frames, where a
   system call was interrupted by a signal.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

#include <execinfo.h>
#include <search.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>

#include "tst-backtrace.h"

#ifndef SIGACTION_FLAGS
# define SIGACTION_FLAGS 0
#endif

/* The backtrace should include at least handle_signal, a signal
   trampoline, read, 3 * fn, and do_test.  */
#define NUM_FUNCTIONS 7

void
handle_signal (int signum)
{
  void *addresses[NUM_FUNCTIONS];
  char **symbols;
  int n;
  int i;

  /* Get the backtrace addresses.  */
  n = backtrace (addresses, sizeof (addresses) / sizeof (addresses[0]));
  printf ("Obtained backtrace with %d functions\n", n);
  /*  Check that there are at least seven functions.  */
  if (n < NUM_FUNCTIONS)
    {
      FAIL ();
      return;
    }
  /* Convert them to symbols.  */
  symbols = backtrace_symbols (addresses, n);
  /* Check that symbols were obtained.  */
  if (symbols == NULL)
    {
      FAIL ();
      return;
    }
  for (i = 0; i < n; ++i)
    printf ("Function %d: %s\n", i, symbols[i]);
  /* Check that the function names obtained are accurate.  */
  if (!match (symbols[0], "handle_signal"))
    {
      FAIL ();
      return;
    }

  /* Do not check name for signal trampoline or cancellable syscall
     wrappers (__syscall_cancel*).  */
  for (; i < n - 1; i++)
    if (match (symbols[i], "read"))
      break;
  if (i == n - 1)
    {
      FAIL ();
      return;
    }

  for (; i < n - 1; i++)
    if (!match (symbols[i], "fn"))
      {
	FAIL ();
	return;
      }
  /* Symbol names are not available for static functions, so we do not
     check do_test.  */

  /* Check that backtrace does not return more than what fits in the array
     (bug 25423).  */
  for (int j = 0; j < NUM_FUNCTIONS; j++)
    {
      n = backtrace (addresses, j);
      if (n > j)
	{
	  FAIL ();
	  return;
	}
    }
}

NO_INLINE int
fn (int c, int flags)
{
  pid_t parent_pid, child_pid;
  int pipefd[2];
  char r[1];
  struct sigaction act;

  if (c > 0)
    {
      fn (c - 1, flags);
      return x;
    }

  memset (&act, 0, sizeof (act));
  act.sa_handler = handle_signal;
  act.sa_flags = flags;
  sigemptyset (&act.sa_mask);
  sigaction (SIGUSR1, &act, NULL);
  parent_pid = getpid ();
  if (pipe (pipefd) == -1)
    abort ();

  child_pid = fork ();
  if (child_pid == (pid_t) -1)
    abort ();
  else if (child_pid == 0)
    {
      sleep (1);
      kill (parent_pid, SIGUSR1);
      _exit (0);
    }

  /* In the parent.  */
  read (pipefd[0], r, 1);

  return 0;
}

NO_INLINE int
do_test (void)
{
  fn (2, SIGACTION_FLAGS);
  return ret;
}

#include <support/test-driver.c>
