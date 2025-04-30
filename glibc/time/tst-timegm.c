/* Basic tests for timegm.
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

static void
do_test_func (time_t (*func)(struct tm *))
{
  {
    struct tm tmg =
      {
	.tm_sec = 0,
	.tm_min = 0,
	.tm_hour = 0,
	.tm_mday = 1,
	.tm_mon = 0,
	.tm_year = 70,
	.tm_wday = 4,
	.tm_yday = 0,
	.tm_isdst = 0
     };
     time_t t = func (&tmg);
     TEST_COMPARE (t, 0);
  }

  {
    struct tm tmg =
      {
	.tm_sec = 7,
	.tm_min = 14,
	.tm_hour = 3,
	.tm_mday = 19,
	.tm_mon = 0,
	.tm_year = 138,
	.tm_wday = 2,
	.tm_yday = 18,
	.tm_isdst = 0
     };
     time_t t = func (&tmg);
     TEST_COMPARE (t, 0x7fffffff);
  }

  if (sizeof (time_t) < 8)
    return;

  {
    struct tm tmg =
      {
	.tm_sec = 8,
	.tm_min = 14,
	.tm_hour = 3,
	.tm_mday = 19,
	.tm_mon = 0,
	.tm_year = 138,
	.tm_wday = 2,
	.tm_yday = 18,
	.tm_isdst = 0
     };
     time_t t = func (&tmg);
     TEST_COMPARE (t, (time_t) 0x80000000ull);
  }
}

static int
do_test (void)
{
  do_test_func (timegm);

  /* timelocal is an alias to mktime and behaves like timegm with the
     difference that it takes timezone into account.  */
  TEST_VERIFY_EXIT (setenv ("TZ", ":", 1) == 0);
  tzset ();
  do_test_func (timelocal);

  return 0;
}

#include <support/test-driver.c>
