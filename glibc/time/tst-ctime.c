/* Test for ctime
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <time.h>
#include <stdlib.h>
#include <support/check.h>

static int
do_test (void)
{
  char *str;
  char strb[32];
  time_t t;

  /* Use glibc time zone extension "TZ=:" to to guarantee that UTC
     without leap seconds is used for the test.  */
  TEST_VERIFY_EXIT (setenv ("TZ", ":", 1) == 0);
  tzset ();

  /* Check if the epoch time can be converted.  */
  t = 0;
  str = ctime (&t);
  TEST_COMPARE_STRING (str, "Thu Jan  1 00:00:00 1970\n");

  /* Same as before but with ctime_r.  */
  str = ctime_r (&t, strb);
  TEST_VERIFY (str == strb);
  TEST_COMPARE_STRING (str, "Thu Jan  1 00:00:00 1970\n");

  /* Check if the max time value for 32 bit time_t can be converted.  */
  t = 0x7fffffff;
  str = ctime (&t);
  TEST_COMPARE_STRING (str, "Tue Jan 19 03:14:07 2038\n");

  /* Same as before but with ctime_r.  */
  str = ctime_r (&t, strb);
  TEST_VERIFY (str == strb);
  TEST_COMPARE_STRING (str, "Tue Jan 19 03:14:07 2038\n");

  /* Check if we run on port with 32 bit time_t size */
  time_t tov;
  if (__builtin_add_overflow (t, 1, &tov))
    return 0;

  /* Check if the time is converted after 32 bit time_t overflow.  */
  str = ctime (&tov);
  TEST_COMPARE_STRING (str, "Tue Jan 19 03:14:08 2038\n");

  /* Same as before but with ctime_r.  */
  str = ctime_r (&tov, strb);
  TEST_VERIFY (str == strb);
  TEST_COMPARE_STRING (str, "Tue Jan 19 03:14:08 2038\n");

  return 0;
}

#include <support/test-driver.c>
