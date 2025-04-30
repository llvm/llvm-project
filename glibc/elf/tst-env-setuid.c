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

/* Verify that tunables correctly filter out unsafe environment variables like
   MALLOC_CHECK_ and MALLOC_MMAP_THRESHOLD_ but also retain
   MALLOC_MMAP_THRESHOLD_ in an unprivileged child.  */

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
#include <support/test-driver.h>
#include <support/capture_subprocess.h>

static char SETGID_CHILD[] = "setgid-child";

#ifndef test_child
static int
test_child (void)
{
  if (getenv ("MALLOC_CHECK_") != NULL)
    {
      printf ("MALLOC_CHECK_ is still set\n");
      return 1;
    }

  if (getenv ("MALLOC_MMAP_THRESHOLD_") == NULL)
    {
      printf ("MALLOC_MMAP_THRESHOLD_ lost\n");
      return 1;
    }

  if (getenv ("LD_HWCAP_MASK") != NULL)
    {
      printf ("LD_HWCAP_MASK still set\n");
      return 1;
    }

  return 0;
}
#endif

#ifndef test_parent
static int
test_parent (void)
{
  if (getenv ("MALLOC_CHECK_") == NULL)
    {
      printf ("MALLOC_CHECK_ lost\n");
      return 1;
    }

  if (getenv ("MALLOC_MMAP_THRESHOLD_") == NULL)
    {
      printf ("MALLOC_MMAP_THRESHOLD_ lost\n");
      return 1;
    }

  if (getenv ("LD_HWCAP_MASK") == NULL)
    {
      printf ("LD_HWCAP_MASK lost\n");
      return 1;
    }

  return 0;
}
#endif

static int
do_test (int argc, char **argv)
{
  /* Setgid child process.  */
  if (argc == 2 && strcmp (argv[1], SETGID_CHILD) == 0)
    {
      if (getgid () == getegid ())
	/* This can happen if the file system is mounted nosuid.  */
	FAIL_UNSUPPORTED ("SGID failed: GID and EGID match (%jd)\n",
			  (intmax_t) getgid ());

      int ret = test_child ();

      if (ret != 0)
	exit (1);

      exit (EXIT_SUCCESS);
    }
  else
    {
      if (test_parent () != 0)
	exit (1);

      int status = support_capture_subprogram_self_sgid (SETGID_CHILD);

      if (WEXITSTATUS (status) == EXIT_UNSUPPORTED)
	return EXIT_UNSUPPORTED;

      if (!WIFEXITED (status))
	FAIL_EXIT1 ("Unexpected exit status %d from child process\n", status);

      return 0;
    }
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
