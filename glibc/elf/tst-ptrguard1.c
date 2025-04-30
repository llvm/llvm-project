/* Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <stackguard-macros.h>
#include <tls.h>
#include <unistd.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
/* Requires _GNU_SOURCE  */
#include <getopt.h>

#ifndef POINTER_CHK_GUARD
extern uintptr_t __pointer_chk_guard;
# define POINTER_CHK_GUARD __pointer_chk_guard
#endif

static const char *command;
static bool child;
static uintptr_t ptr_chk_guard_copy;
static bool ptr_chk_guard_copy_set;
static int fds[2];

static void __attribute__ ((constructor))
con (void)
{
  ptr_chk_guard_copy = POINTER_CHK_GUARD;
  ptr_chk_guard_copy_set = true;
}

static int
uintptr_t_cmp (const void *a, const void *b)
{
  if (*(uintptr_t *) a < *(uintptr_t *) b)
    return 1;
  if (*(uintptr_t *) a > *(uintptr_t *) b)
    return -1;
  return 0;
}

static int
do_test (void)
{
  if (!ptr_chk_guard_copy_set)
    {
      puts ("constructor has not been run");
      return 1;
    }

  if (ptr_chk_guard_copy != POINTER_CHK_GUARD)
    {
      puts ("POINTER_CHK_GUARD changed between constructor and do_test");
      return 1;
    }

  if (child)
    {
      write (2, &ptr_chk_guard_copy, sizeof (ptr_chk_guard_copy));
      return 0;
    }

  if (command == NULL)
    {
      puts ("missing --command or --child argument");
      return 1;
    }

#define N 16
  uintptr_t child_ptr_chk_guards[N + 1];
  child_ptr_chk_guards[N] = ptr_chk_guard_copy;
  int i;
  for (i = 0; i < N; ++i)
    {
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
	  if (ptr_chk_guard_copy != POINTER_CHK_GUARD)
	    {
	      puts ("POINTER_CHK_GUARD changed after fork");
	      exit (1);
	    }

	  close (fds[0]);
	  close (2);
	  dup2 (fds[1], 2);
	  close (fds[1]);

	  system (command);
	  exit (0);
	}

      close (fds[1]);

      if (TEMP_FAILURE_RETRY (read (fds[0], &child_ptr_chk_guards[i],
				    sizeof (uintptr_t))) != sizeof (uintptr_t))
	{
	  puts ("could not read ptr_chk_guard value from child");
	  return 1;
	}

      close (fds[0]);

      pid_t termpid;
      int status;
      termpid = TEMP_FAILURE_RETRY (waitpid (pid, &status, 0));
      if (termpid == -1)
	{
	  printf ("waitpid failed: %m\n");
	  return 1;
	}
      else if (termpid != pid)
	{
	  printf ("waitpid returned %ld != %ld\n",
		  (long int) termpid, (long int) pid);
	  return 1;
	}
      else if (!WIFEXITED (status) || WEXITSTATUS (status))
	{
	  puts ("child hasn't exited with exit status 0");
	  return 1;
	}
    }

  qsort (child_ptr_chk_guards, N + 1, sizeof (uintptr_t), uintptr_t_cmp);

  /* The default pointer guard is the same as the default stack guard.
     They are only set to default if dl_random is NULL.  */
  uintptr_t default_guard = 0;
  unsigned char *p = (unsigned char *) &default_guard;
  p[sizeof (uintptr_t) - 1] = 255;
  p[sizeof (uintptr_t) - 2] = '\n';
  p[0] = 0;

  /* Test if the pointer guard canaries are either randomized,
     or equal to the default pointer guard value.
     Even with randomized pointer guards it might happen
     that the random number generator generates the same
     values, but if that happens in more than half from
     the 16 runs, something is very wrong.  */
  int ndifferences = 0;
  int ndefaults = 0;
  for (i = 0; i < N; ++i)
    {
      if (child_ptr_chk_guards[i] != child_ptr_chk_guards[i+1])
	ndifferences++;
      else if (child_ptr_chk_guards[i] == default_guard)
	ndefaults++;
    }

  printf ("differences %d defaults %d\n", ndifferences, ndefaults);

  if (ndifferences < N / 2 && ndefaults < N / 2)
    {
      puts ("pointer guard values are not randomized enough");
      puts ("nor equal to the default value");
      return 1;
    }

  return 0;
}

#define OPT_COMMAND	10000
#define OPT_CHILD	10001
#define CMDLINE_OPTIONS	\
  { "command", required_argument, NULL, OPT_COMMAND },  \
  { "child", no_argument, NULL, OPT_CHILD },

static void __attribute((used))
cmdline_process_function (int c)
{
  switch (c)
    {
      case OPT_COMMAND:
        command = optarg;
        break;
      case OPT_CHILD:
        child = true;
        break;
    }
}

#define CMDLINE_PROCESS	cmdline_process_function

#include <support/test-driver.c>
