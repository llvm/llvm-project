/* Test for vfork functions.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

/* This test relies on non-POSIX functionality since the child
   processes call write and getpid.  */
static int
do_test (void)
{
  int result = 0;
  int fd[2];

  if (pipe (fd) == -1)
    {
      puts ("pipe failed");
      return 1;
    }

  /* First vfork() without previous getpid().  */
  pid_t p1;
  if ((p1 = vfork ()) == 0)
    {
      pid_t p = getpid ();
      _exit (TEMP_FAILURE_RETRY (write (fd[1], &p, sizeof (p))) != sizeof (p));
    }
  else if (p1 == -1)
    {
      puts ("1st vfork failed");
      result = 1;
    }

  pid_t p2 = 0;
  if (TEMP_FAILURE_RETRY (read (fd[0], &p2, sizeof (pid_t))) != sizeof (pid_t))
    {
      puts ("1st read failed");
      result = 1;
    }
  int r;
  if (TEMP_FAILURE_RETRY (waitpid (p1, &r, 0)) != p1)
    {
      puts ("1st waitpid failed");
      result = 1;
    }
  else if (r != 0)
    {
      puts ("write in 1st child failed");
      result = 1;
    }

  /* Main process' ID.  */
  pid_t p0 = getpid ();

  /* vfork() again, but after a getpid() in the main process.  */
  pid_t p3;
  if ((p3 = vfork ()) == 0)
    {
      pid_t p = getpid ();
      _exit (TEMP_FAILURE_RETRY (write (fd[1], &p, sizeof (p))) != sizeof (p));
    }
  else if (p1 == -1)
    {
      puts ("2nd vfork failed");
      result = 1;
    }

  pid_t p4;
  if (TEMP_FAILURE_RETRY (read (fd[0], &p4, sizeof (pid_t))) != sizeof (pid_t))
    {
      puts ("2nd read failed");
      result = 1;
    }
  if (TEMP_FAILURE_RETRY (waitpid (p3, &r, 0)) != p3)
    {
      puts ("2nd waitpid failed");
      result = 1;
    }
  else if (r != 0)
    {
      puts ("write in 2nd child failed");
      result = 1;
    }

  /* And getpid in the main process again.  */
  pid_t p5 = getpid ();

  /* Analysis of the results.  */
  if (p0 != p5)
    {
      printf ("p0(%ld) != p5(%ld)\n", (long int) p0, (long int) p5);
      result = 1;
    }

  if (p0 == p1)
    {
      printf ("p0(%ld) == p1(%ld)\n", (long int) p0, (long int) p1);
      result = 1;
    }

  if (p1 != p2)
    {
      printf ("p1(%ld) != p2(%ld)\n", (long int) p1, (long int) p2);
      result = 1;
    }

  if (p0 == p3)
    {
      printf ("p0(%ld) == p3(%ld)\n", (long int) p0, (long int) p3);
      result = 1;
    }

  if (p3 != p4)
    {
      printf ("p3(%ld) != p4(%ld)\n", (long int) p3, (long int) p4);
      result = 1;
    }

  close (fd[0]);
  close (fd[1]);

  if (result == 0)
    puts ("All OK");

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
