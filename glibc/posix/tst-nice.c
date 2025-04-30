/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <unistd.h>


/* Test that nice() does not incorrectly return 0.  */
static int
do_test (void)
{
  int ret;
  const int incr = 10;
  int old;

  /* Discover current nice value.  */
  errno = 0;
  old = nice (0);
  if (old == -1 && errno != 0)
    {
      printf ("break: nice(%d) return: %d, %m\n", 0, old);
      return 1;
    }

  /* Nice ourselves up.  */
  errno = 0;
  ret = nice (incr);
  if (ret == -1 && errno != 0)
    {
      printf ("break: nice(%d) return: %d, %m\n", incr, ret);
      return 1;
    }

  /* Check for return value being zero when it shouldn't.  Cannot simply
     check for expected value since nice values are capped at 2^n-1.
     But we assume that we didn't start at the cap and so should have
     increased some.  */
  if (ret <= old)
    {
      printf ("FAIL: retval (%d) of nice(%d) != %d\n", ret, incr, old + incr);
      return 1;
    }

  /* BZ #18086. Make sure we don't reset errno.  */
  errno = EBADF;
  nice (0);
  if (errno != EBADF)
    {
      printf ("FAIL: errno = %i, but wanted EBADF (%i)\n", errno, EBADF);
      return 1;
    }


  printf ("PASS: nice(%d) from %d return: %d\n", incr, old, ret);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
