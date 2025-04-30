/* Test for mktime (4)
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

const struct tm tm0 =
  {
    .tm_year = 70,
    .tm_mon = 0,
    .tm_mday = 1,
    .tm_hour = 0,
    .tm_min = 0,
    .tm_sec = 0,
    .tm_wday = 4,
    .tm_yday = 0,
  };

const struct tm tmY2038 =
  {
    .tm_year = 138,
    .tm_mon = 0,
    .tm_mday = 19,
    .tm_hour = 3,
    .tm_min = 14,
    .tm_sec = 7,
  };

const struct tm tm32bitmax =
  {
    .tm_year = 206,
    .tm_mon = 1,
    .tm_mday = 7,
    .tm_hour = 6,
    .tm_min = 28,
    .tm_sec = 15,
  };

static
int test_mktime_helper (struct tm *tm, long long int exp_val, int line)
{
  time_t result, t;

  /* Check if we run on port with 32 bit time_t size.  */
  if (__builtin_add_overflow (exp_val, 0, &t))
    return 0;

  result = mktime (tm);
  if (result == (time_t) -1)
    FAIL_RET ("*** mktime failed: %m in line: %d", line);

  if ((long long int) result != exp_val)
    FAIL_RET ("*** Result different than expected (%lld != %lld) in %d\n",
              (long long int) result, exp_val, line);

  return 0;
}

static int
do_test (void)
{
  struct tm t;
  /* Use glibc time zone extension "TZ=:" to to guarantee that UTC
     without leap seconds is used for the test.  */
  TEST_VERIFY_EXIT (setenv ("TZ", ":", 1) == 0);

  /* Check that mktime (1970-01-01 00:00:00) returns 0.  */
  t = tm0;
  test_mktime_helper (&t, 0, __LINE__);

  /* Check that mktime (2038-01-19 03:14:07) returns 0x7FFFFFFF.  */
  t = tmY2038;
  test_mktime_helper (&t, 0x7fffffff, __LINE__);

  /* Check that mktime (2038-01-19 03:14:08) returns 0x80000000
     (time_t overflow).  */
  t = tmY2038;
  t.tm_sec++;
  test_mktime_helper (&t, 0x80000000, __LINE__);

  /* Check that mktime (2106-02-07 06:28:15) returns 0xFFFFFFFF.  */
  t = tm32bitmax;
  test_mktime_helper (&t, 0xFFFFFFFF, __LINE__);

  /* Check that mktime (2106-02-07 06:28:16) returns 0x100000000.  */
  t = tm32bitmax;
  t.tm_sec++;
  test_mktime_helper (&t, 0x100000000, __LINE__);

  return 0;
}

#include <support/test-driver.c>
