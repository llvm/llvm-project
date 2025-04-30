/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* Verify that tunables correctly filter out unsafe tunables like
   glibc.malloc.check and glibc.malloc.mmap_threshold but also retain
   glibc.malloc.mmap_threshold in an unprivileged child.  */

/* This is compiled as part of the testsuite but needs to see
   HAVE_TUNABLES. */
#define _LIBC 1
#include "config.h"
#undef _LIBC

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <intprops.h>
#include <array_length.h>

#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/capture_subprocess.h>

const char *teststrings[] =
{
  "glibc.malloc.check=2:glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.check=2:glibc.malloc.check=2:glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.check=2:glibc.malloc.mmap_threshold=4096:glibc.malloc.check=2",
  "glibc.malloc.perturb=0x800",
  "glibc.malloc.perturb=0x800:glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.perturb=0x800:not_valid.malloc.check=2:glibc.malloc.mmap_threshold=4096",
  "glibc.not_valid.check=2:glibc.malloc.mmap_threshold=4096",
  "not_valid.malloc.check=2:glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.garbage=2:glibc.maoc.mmap_threshold=4096:glibc.malloc.check=2",
  "glibc.malloc.check=4:glibc.malloc.garbage=2:glibc.maoc.mmap_threshold=4096",
  ":glibc.malloc.garbage=2:glibc.malloc.check=1",
  "glibc.malloc.check=1:glibc.malloc.check=2",
  "not_valid.malloc.check=2",
  "glibc.not_valid.check=2",
};

const char *resultstrings[] =
{
  "glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.perturb=0x800",
  "glibc.malloc.perturb=0x800:glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.perturb=0x800:glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.mmap_threshold=4096",
  "glibc.malloc.mmap_threshold=4096",
  "",
  "",
  "",
  "",
  "",
  "",
};

static int
test_child (int off)
{
  const char *val = getenv ("GLIBC_TUNABLES");

#if HAVE_TUNABLES
  if (val != NULL && strcmp (val, resultstrings[off]) == 0)
    return 0;

  if (val != NULL)
    printf ("[%d] Unexpected GLIBC_TUNABLES VALUE %s\n", off, val);

  return 1;
#else
  if (val != NULL)
    {
      printf ("[%d] GLIBC_TUNABLES not cleared\n", off);
      return 1;
    }
  return 0;
#endif
}

static int
do_test (int argc, char **argv)
{
  /* Setgid child process.  */
  if (argc == 2)
    {
      if (getgid () == getegid ())
	/* This can happen if the file system is mounted nosuid.  */
	FAIL_UNSUPPORTED ("SGID failed: GID and EGID match (%jd)\n",
			  (intmax_t) getgid ());

      int ret = test_child (atoi (argv[1]));

      if (ret != 0)
	exit (1);

      exit (EXIT_SUCCESS);
    }
  else
    {
      int ret = 0;

      /* Spawn tests.  */
      for (int i = 0; i < array_length (teststrings); i++)
	{
	  char buf[INT_BUFSIZE_BOUND (int)];

	  printf ("Spawned test for %s (%d)\n", teststrings[i], i);
	  snprintf (buf, sizeof (buf), "%d\n", i);
	  if (setenv ("GLIBC_TUNABLES", teststrings[i], 1) != 0)
	    exit (1);

	  int status = support_capture_subprogram_self_sgid (buf);

	  /* Bail out early if unsupported.  */
	  if (WEXITSTATUS (status) == EXIT_UNSUPPORTED)
	    return EXIT_UNSUPPORTED;

	  ret |= status;
	}
      return ret;
    }
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
