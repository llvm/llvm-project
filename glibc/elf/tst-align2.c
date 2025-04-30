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

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <tst-stack-align.h>
#include <unistd.h>

static int res, fds[2], result;
static bool test_destructors;

extern void in_dso (int *, bool *, int *);

static void __attribute__ ((constructor)) con (void)
{
  res = TEST_STACK_ALIGN () ? -1 : 1;
}

static void __attribute__ ((destructor)) des (void)
{
  if (!test_destructors)
    return;

  char c = TEST_STACK_ALIGN () ? 'B' : 'A';
  write (fds[1], &c, 1);
}

static int
do_test (void)
{
  if (!res)
    {
      puts ("binary's constructor has not been run");
      result = 1;
    }
  else if (res != 1)
    {
      puts ("binary's constructor has been run without sufficient alignment");
      result = 1;
    }

  if (TEST_STACK_ALIGN ())
    {
      puts ("insufficient stack alignment in do_test");
      result = 1;
    }

  in_dso (&result, &test_destructors, &fds[1]);

  if (pipe (fds) < 0)
    {
      printf ("couldn't create pipe: %m\n");
      return 1;
    }

  pid_t pid = fork ();
  if (pid < 0)
    {
      printf ("fork failed: %m\n");
      return 1;
    }

  if (!pid)
    {
      close (fds[0]);
      test_destructors = true;
      exit (0);
    }

  close (fds[1]);

  unsigned char c;
  ssize_t len;
  int des_seen = 0, dso_des_seen = 0;
  while ((len = TEMP_FAILURE_RETRY (read (fds[0], &c, 1))) > 0)
    {
      switch (c)
        {
        case 'B':
          puts ("insufficient alignment in binary's destructor");
          result = 1;
          /* FALLTHROUGH */
        case 'A':
          des_seen++;
          break;
        case 'D':
          puts ("insufficient alignment in DSO destructor");
          result = 1;
          /* FALLTHROUGH */
        case 'C':
          dso_des_seen++;
          break;
        default:
          printf ("unexpected character %x read from pipe", c);
          result = 1;
          break;
        }
    }

  close (fds[0]);

  if (des_seen != 1)
    {
      printf ("binary destructor run %d times instead of once\n", des_seen);
      result = 1;
    }

  if (dso_des_seen != 1)
    {
      printf ("DSO destructor run %d times instead of once\n", dso_des_seen);
      result = 1;
    }

  int status;
  pid_t termpid;
  termpid = TEMP_FAILURE_RETRY (waitpid (pid, &status, 0));
  if (termpid == -1)
    {
      printf ("waitpid failed: %m\n");
      result = 1;
    }
  else if (termpid != pid)
    {
      printf ("waitpid returned %ld != %ld\n",
	      (long int) termpid, (long int) pid);
      result = 1;
    }
  else if (!WIFEXITED (status) || WEXITSTATUS (status))
    {
      puts ("child hasn't exited with exit status 0");
      result = 1;
    }

  return result;
}

#include <support/test-driver.c>
