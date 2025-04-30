/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Test that secure_getenv works by invoking the test as a SGID
   program with a group ID from the supplementary group list.  This
   test can fail spuriously if the user is not a member of a suitable
   supplementary group.  */

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <support/check.h>
#include <support/support.h>
#include <support/capture_subprocess.h>
#include <support/test-driver.h>

static char MAGIC_ARGUMENT[] = "run-actual-test";

static int
do_test (void)
{
  if (getenv ("PATH") == NULL)
    {
      printf ("PATH not set\n");
      exit (1);
    }
  if (secure_getenv ("PATH") == NULL)
    {
      printf ("PATH not set according to secure_getenv\n");
      exit (1);
    }
  if (strcmp (getenv ("PATH"), secure_getenv ("PATH")) != 0)
    {
      printf ("PATH mismatch (%s, %s)\n",
	      getenv ("PATH"), secure_getenv ("PATH"));
      exit (1);
    }

  int status = support_capture_subprogram_self_sgid (MAGIC_ARGUMENT);

  if (WEXITSTATUS (status) == EXIT_UNSUPPORTED)
    return EXIT_UNSUPPORTED;

  if (!WIFEXITED (status))
    FAIL_EXIT1 ("Unexpected exit status %d from child process\n", status);

  return 0;
}

static void
alternative_main (int argc, char **argv)
{
  if (argc == 2 && strcmp (argv[1], MAGIC_ARGUMENT) == 0)
    {
      if (getgid () == getegid ())
	/* This can happen if the file system is mounted nosuid.  */
	FAIL_UNSUPPORTED ("SGID failed: GID and EGID match (%jd)\n",
		   (intmax_t) getgid ());
      if (getenv ("PATH") == NULL)
	FAIL_EXIT (3, "PATH variable not present\n");
      if (secure_getenv ("PATH") != NULL)
	FAIL_EXIT (4, "PATH variable not filtered out\n");

      exit (EXIT_SUCCESS);
    }
}

#define PREPARE alternative_main
#include <support/test-driver.c>
