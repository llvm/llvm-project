/* Test for localtime bug in year 2039 (bug 22639).
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <string.h>
#include <support/check.h>

static int
do_test (void)
{
  TEST_VERIFY_EXIT (setenv ("TZ", "PST8PDT,M3.2.0,M11.1.0", 1) == 0);
  tzset ();
  if (sizeof (time_t) > 4)
    {
      time_t ouch = (time_t) 2187810000LL;
      char buf[500];
      struct tm *tm = localtime (&ouch);
      TEST_VERIFY_EXIT (tm != NULL);
      TEST_VERIFY_EXIT (strftime (buf, sizeof buf, "%Y-%m-%d %H:%M:%S %Z", tm)
			> 0);
      puts (buf);
      TEST_VERIFY (strcmp (buf, "2039-04-30 14:00:00 PDT") == 0);

      /* Same as before but for localtime_r.  */
      struct tm tmd;
      tm = localtime_r (&ouch, &tmd);
      TEST_VERIFY_EXIT (tm == &tmd);

      TEST_VERIFY_EXIT (strftime (buf, sizeof buf, "%Y-%m-%d %H:%M:%S %Z", tm)
			> 0);
      puts (buf);
      TEST_VERIFY (strcmp (buf, "2039-04-30 14:00:00 PDT") == 0);
    }
  else
    FAIL_UNSUPPORTED ("32-bit time_t");
  return 0;
}

#include <support/test-driver.c>
