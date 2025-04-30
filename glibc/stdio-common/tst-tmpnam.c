/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
do_test (void)
{
  const char *name;
  int retval = 0;

  /* Set TMPDIR to a value other than the traditional /tmp.  */
  setenv ("TMPDIR", "/usr", 1);

  name = tmpnam (NULL);

  printf ("name = %s\n", name);

  /* Make sure the name is not based on the value in TMPDIR.  */
  if (strncmp (name, "/usr", 4) == 0)
    {
      puts ("error: `tmpnam' used TMPDIR value");
      retval = 1;
    }

  /* Test that it is in the directory denoted by P_tmpdir.  */
  if (strncmp (name, P_tmpdir, sizeof (P_tmpdir) - 1) != 0)
    {
      puts ("error: `tmpnam' return value not in P_tmpdir directory");
      retval = 1;
    }

  return retval;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
