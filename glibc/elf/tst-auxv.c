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

#include <elf.h>
#include <errno.h>
#include <link.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <misc/sys/auxv.h>

static int
do_test (int argc, char *argv[])
{
  errno = 0;
  const char *execfn = (const char *) getauxval (AT_NULL);

  if (errno != ENOENT)
    {
      printf ("errno is %d rather than %d (ENOENT) on failure\n", errno,
	      ENOENT);
      return 1;
    }

  if (execfn != NULL)
    {
      printf ("getauxval return value is nonzero on failure\n");
      return 1;
    }

  errno = 0;
  execfn = (const char *) getauxval (AT_EXECFN);

  if (execfn == NULL)
    {
      printf ("No AT_EXECFN found, AT_EXECFN test skipped\n");
      return 0;
    }

  if (errno != 0)
    {
      printf ("errno erroneously set to %d on success\n", errno);
      return 1;
    }

  if (strcmp (argv[0], execfn) != 0)
    {
      printf ("Mismatch: argv[0]: %s vs. AT_EXECFN: %s\n", argv[0], execfn);
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
