/* Verify that print functions return error when there is an I/O error.

   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


static int
do_test (void)
{
  char tmpl[] = "/tmp/tst-put-error.XXXXXX";
  int fd = mkstemp (tmpl);
  if (fd == -1)
    error (EXIT_FAILURE, errno, "cannot create temporary file");
  FILE *fp = fdopen (fd, "w");
  if (fp == NULL)
    error (EXIT_FAILURE, errno, "fdopen");

  /* All of the tests below verify that flushing buffers result in failure of
     the fprintf calls.  We ensure that the buffer is flushed at the end of
     each fprintf call by doing two things - setting the file pointer to
     line-buffered so that it is flushed whenever it encounters a newline and
     then ensuring that there is a newline in each of the format strings we
     pass to fprintf.  */

  setlinebuf (fp);
  close (fd);
  unlink (tmpl);

  int n = fprintf (fp, "hello world\n");
  printf ("fprintf = %d\n", n);
  if (n >= 0)
    error (EXIT_FAILURE, 0, "first fprintf succeeded");

  n = fprintf (fp, "hello world\n");
  printf ("fprintf = %d\n", n);
  if (n >= 0)
    error (EXIT_FAILURE, 0, "second fprintf succeeded");

  /* Padded printing takes a different code path.  */
  n = fprintf (fp, "%100s\n", "foo");
  printf ("fprintf = %d\n", n);
  if (n >= 0)
    error (EXIT_FAILURE, 0, "padded fprintf succeeded");

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
