/* Test for bug 19439.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Marek Polacek <polacek@redhat.com>, 2012.

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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  /* Verify that isinff, isinfl, isnanf, and isnanlf are defined
     in the header under C++11 and can be called.  Without the
     header fix this test will not compile.  */
  if (isinff (1.0f)
      || !isinff (INFINITY)
      || isinfl (1.0L)
      || !isinfl (INFINITY)
      || isnanf (2.0f)
      || !isnanf (NAN)
      || isnanl (2.0L)
      || !isnanl (NAN)
      )
    {
      printf ("FAIL: Failed to call is* functions.\n");
      exit (1);
    }
  printf ("PASS: Able to call isinff, isinfl, isnanf, and isnanl.\n");
  exit (0);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
