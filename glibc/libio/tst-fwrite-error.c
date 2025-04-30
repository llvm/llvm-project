/* Test of fwrite() function, adapted from gnulib-tests in grep.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.

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
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  char tmpl[] = "/tmp/tst-fwrite-error.XXXXXX";
  int fd = mkstemp (tmpl);
  if (fd == -1)
    {
      printf ("mkstemp failed with errno %d\n", errno);
      return 1;
    }
  FILE *fp = fdopen (fd, "w");
  if (fp == NULL)
    {
      printf ("fdopen failed with errno %d\n", errno);
      return 1;
    }

  char buf[] = "world";
  setvbuf (fp, NULL, _IONBF, 0);
  close (fd);
  unlink (tmpl);
  errno = 0;

  int ret = fwrite (buf, 1, sizeof (buf), fp);
  if (ret != 0)
    {
      printf ("fwrite returned %d\n", ret);
      return 1;
    }
  if (errno != EBADF)
    {
      printf ("Errno is not EBADF: %d\n", errno);
      return 1;
    }
  if (ferror (fp) == 0)
    {
      printf ("ferror not set\n");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
