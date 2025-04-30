/* Basic tests for gmtime.
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
  /* Check if the epoch time can be converted.  */
  {
    time_t t = 0;
    struct tm *tmg = gmtime (&t);
    TEST_COMPARE (tmg->tm_sec,   0);
    TEST_COMPARE (tmg->tm_min,   0);
    TEST_COMPARE (tmg->tm_hour,  0);
    TEST_COMPARE (tmg->tm_mday,  1);
    TEST_COMPARE (tmg->tm_mon,   0);
    TEST_COMPARE (tmg->tm_year,  70);
    TEST_COMPARE (tmg->tm_wday,  4);
    TEST_COMPARE (tmg->tm_yday,  0);
    TEST_COMPARE (tmg->tm_isdst, 0);
  }
  {
    /* Same as before but with gmtime_r.  */
    time_t t = 0;
    struct tm tms;
    struct tm *tmg = gmtime_r (&t, &tms);
    TEST_VERIFY (tmg == &tms);
    TEST_COMPARE (tmg->tm_sec,   0);
    TEST_COMPARE (tmg->tm_min,   0);
    TEST_COMPARE (tmg->tm_hour,  0);
    TEST_COMPARE (tmg->tm_mday,  1);
    TEST_COMPARE (tmg->tm_mon,   0);
    TEST_COMPARE (tmg->tm_year,  70);
    TEST_COMPARE (tmg->tm_wday,  4);
    TEST_COMPARE (tmg->tm_yday,  0);
    TEST_COMPARE (tmg->tm_isdst, 0);
  }

  /* Check if the max time value for 32 bit time_t can be converted.  */
  {
    time_t t = 0x7fffffff;
    struct tm *tmg = gmtime (&t);
    TEST_COMPARE (tmg->tm_sec,   7);
    TEST_COMPARE (tmg->tm_min,   14);
    TEST_COMPARE (tmg->tm_hour,  3);
    TEST_COMPARE (tmg->tm_mday,  19);
    TEST_COMPARE (tmg->tm_mon,   0);
    TEST_COMPARE (tmg->tm_year,  138);
    TEST_COMPARE (tmg->tm_wday,  2);
    TEST_COMPARE (tmg->tm_yday,  18);
    TEST_COMPARE (tmg->tm_isdst, 0);
  }
  {
    /* Same as before but with ctime_r.  */
    time_t t = 0x7fffffff;
    struct tm tms;
    struct tm *tmg = gmtime_r (&t, &tms);
    TEST_VERIFY (tmg == &tms);
    TEST_COMPARE (tmg->tm_sec,   7);
    TEST_COMPARE (tmg->tm_min,   14);
    TEST_COMPARE (tmg->tm_hour,  3);
    TEST_COMPARE (tmg->tm_mday,  19);
    TEST_COMPARE (tmg->tm_mon,   0);
    TEST_COMPARE (tmg->tm_year,  138);
    TEST_COMPARE (tmg->tm_wday,  2);
    TEST_COMPARE (tmg->tm_yday,  18);
    TEST_COMPARE (tmg->tm_isdst, 0);
  }

  if (sizeof (time_t) < 8)
    return 0;

  {
    time_t t = (time_t) 0x80000000ull;
    struct tm *tmg = gmtime (&t);
    TEST_COMPARE (tmg->tm_sec,   8);
    TEST_COMPARE (tmg->tm_min,   14);
    TEST_COMPARE (tmg->tm_hour,  3);
    TEST_COMPARE (tmg->tm_mday,  19);
    TEST_COMPARE (tmg->tm_mon,   0);
    TEST_COMPARE (tmg->tm_year,  138);
    TEST_COMPARE (tmg->tm_wday,  2);
    TEST_COMPARE (tmg->tm_yday,  18);
    TEST_COMPARE (tmg->tm_isdst, 0);
  }

  {
    time_t t = (time_t) 0x80000000ull;
    struct tm tms;
    struct tm *tmg = gmtime_r (&t, &tms);
    TEST_VERIFY (tmg == &tms);
    TEST_COMPARE (tmg->tm_sec,   8);
    TEST_COMPARE (tmg->tm_min,   14);
    TEST_COMPARE (tmg->tm_hour,  3);
    TEST_COMPARE (tmg->tm_mday,  19);
    TEST_COMPARE (tmg->tm_mon,   0);
    TEST_COMPARE (tmg->tm_year,  138);
    TEST_COMPARE (tmg->tm_wday,  2);
    TEST_COMPARE (tmg->tm_yday,  18);
    TEST_COMPARE (tmg->tm_isdst, 0);
  }

  return 0;
}

#include <support/test-driver.c>
