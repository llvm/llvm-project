/* Verify that child of the vfork-calling function can't return when
   shadow stack is in use.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <x86intrin.h>
#include <support/test-driver.h>
#include <support/xsignal.h>
#include <support/xunistd.h>

__attribute__ ((noclone, noinline))
static void
do_test_1 (void)
{
  pid_t p1;
  int fd[2];

  if (pipe (fd) == -1)
    {
      puts ("pipe failed");
      _exit (EXIT_FAILURE);
    }

  if ((p1 = vfork ()) == 0)
    {
      pid_t p = getpid ();
      TEMP_FAILURE_RETRY (write (fd[1], &p, sizeof (p)));
      /* Child return should trigger SIGSEGV.  */
      return;
    }
  else if (p1 == -1)
    {
      puts ("vfork failed");
      _exit (EXIT_FAILURE);
    }

  pid_t p2 = 0;
  if (TEMP_FAILURE_RETRY (read (fd[0], &p2, sizeof (pid_t)))
      != sizeof (pid_t))
    puts ("pipd read failed");
  else
    {
      int r;
      if (TEMP_FAILURE_RETRY (waitpid (p1, &r, 0)) != p1)
	puts ("waitpid failed");
      else if (r != 0)
	puts ("pip write in child failed");
    }

  /* Parent exits immediately so that parent returns without triggering
     SIGSEGV when shadow stack isn't in use.  */
  _exit (EXIT_FAILURE);
}

static int
do_test (void)
{
  /* NB: This test should trigger SIGSEGV with shadow stack enabled.  */
  if (_get_ssp () == 0)
    return EXIT_UNSUPPORTED;
  do_test_1 ();
  /* Child exits immediately so that child returns without triggering
     SIGSEGV when shadow stack isn't in use.  */
  _exit (EXIT_FAILURE);
}

#define EXPECTED_SIGNAL (_get_ssp () == 0 ? 0 : SIGSEGV)
#include <support/test-driver.c>
