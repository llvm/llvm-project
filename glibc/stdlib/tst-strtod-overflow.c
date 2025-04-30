/* Test for integer/buffer overflow in strtod.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <support/blob_repeat.h>
#include <support/test-driver.h>

#define EXPONENT "e-2147483649"
#define SIZE 214748364

static int
do_test (void)
{
  struct support_blob_repeat repeat = support_blob_repeat_allocate
    ("0", 1, 1 + SIZE + sizeof (EXPONENT));
  if (repeat.size == 0)
    {
      puts ("warning: memory allocation failed, cannot test for overflow");
      return EXIT_UNSUPPORTED;
    }
  char *p = repeat.start;
  p[0] = '1';
  memcpy (p + 1 + SIZE, EXPONENT, sizeof (EXPONENT));
  double d = strtod (p, NULL);
  if (d != 0)
    {
      printf ("error: strtod returned wrong value: %a\n", d);
      return 1;
    }
  support_blob_repeat_free (&repeat);
  return 0;
}

#define TEST_FUNCTION do_test ()
#define TIMEOUT 30
#include "../test-skeleton.c"
