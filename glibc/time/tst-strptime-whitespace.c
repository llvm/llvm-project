/* Verify that strptime accepts arbitrary whitespace between tokens.

   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#define _XOPEN_SOURCE
#include <time.h>
#include <stdio.h>
#include <string.h>

int
do_test (void)
{
  struct tm t;
  const char *in = "Tuesday \t 22 \t July\t1942";

  char *r = strptime (in, "%A%d %b%Y", &t);

  if (r == NULL || r != in + strlen (in))
    {
      printf ("strptime failed\n");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
