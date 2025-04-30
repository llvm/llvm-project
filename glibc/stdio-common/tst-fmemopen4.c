/* fmemopen tests for BZ#1930 and BZ#20005.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>


/* Check if fflush does not reset the file position.  */
static int
do_test (void)
{
  char buffer[500] = "x";

  FILE *stream = fmemopen (buffer, sizeof (buffer), "r+");
  if (stream == NULL)
    {
      printf ("error: fmemopen could not open stream\n");
      return 1;
    }

  const char test[] = "test";

  size_t r = fwrite (test, sizeof (char), sizeof (test), stream);
  if (r != sizeof (test))
    {
      printf ("error: fwrite returned %zu, expected %zu\n", r, sizeof (test));
      return 1;
    }

  r = ftell (stream);
  if (r != sizeof (test))
    {
      printf ("error: ftell return %zu, expected %zu\n", r, sizeof (test));
      return 1;
    }

  if (fflush (stream) != 0)
    {
      printf ("error: fflush failed\n");
      return 1;
    }

  r = ftell (stream);
  if (r != sizeof (test))
    {
      printf ("error: ftell return %zu, expected %zu\n", r, sizeof (test));
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
