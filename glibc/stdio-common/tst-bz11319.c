/* Regression test for bug 11319.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define _GNU_SOURCE 1

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

#include <support/check.h>
#include <support/temp_file.h>
#include <support/xunistd.h>

static int
do_test (void)
{
  char *tempfile;
  int fd;

  /* Create a temporary file and open it in read-only mode.  */
  TEST_VERIFY_EXIT (create_temp_file ("tst-bz11319", &tempfile));
  fd = xopen (tempfile, O_RDONLY, 0660);

  /* Try and write to the temporary file to intentionally fail, then
     check that dprintf (or __dprintf_chk) return EOF.  */
  TEST_COMPARE (dprintf (fd, "%d", 0), EOF);

  xclose (fd);
  free (tempfile);

  return 0;
}

#include <support/test-driver.c>
