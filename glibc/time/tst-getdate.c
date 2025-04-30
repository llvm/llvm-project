/* Test for getdate.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2000.

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

#include <array_length.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/temp_file.h>
#include <support/xunistd.h>
#include <time.h>

static const struct
{
  const char *str;
  const char *tz;
  struct tm tm;
  bool time64;
} tests [] =
{
  {"21:01:10 1999-1-31", "Universal", {10, 1, 21, 31, 0, 99, 0, 0, 0},
   false },
  {"21:01:10    1999-1-31", "Universal", {10, 1, 21, 31, 0, 99, 0, 0, 0},
   false },
  {"   21:01:10 1999-1-31", "Universal", {10, 1, 21, 31, 0, 99, 0, 0, 0},
   false },
  {"21:01:10 1999-1-31   ", "Universal", {10, 1, 21, 31, 0, 99, 0, 0, 0},
   false },
  {"    21:01:10 1999-1-31   ", "Universal", {10, 1, 21, 31, 0, 99, 0, 0, 0},
   false },
  {"21:01:10 1999-2-28", "Universal", {10, 1, 21, 28, 1, 99, 0, 0, 0},
   false },
  {"16:30:46 2000-2-29", "Universal", {46, 30,16, 29, 1, 100, 0, 0, 0},
   false },
  {"01-08-2000 05:06:07", "Europe/Berlin", {7, 6, 5, 1, 7, 100, 0, 0, 0},
   false },

  /* 64 bit time_t tests.  */
  {"21:01:10 2038-1-31", "Universal", {10, 1, 21, 31, 0, 138, 0, 0, 0},
   true },
  {"22:01:10 2048-5-20", "Universal", {10, 1, 22, 20, 4, 148, 0, 0, 0},
   true },
  {"01-08-2038 05:06:07", "Europe/Berlin", {7, 6, 5, 1, 7, 138, 0, 0, 0},
   true },
  {"20-03-2050 21:30:08", "Europe/Berlin", {8, 30, 21, 20, 2, 150, 0, 0, 0},
   true }
};

static const char *
report_date_error (void)
{
  switch (getdate_err)
    {
    case 1:
      return "The environment variable DATEMSK is not defined or null.";
    case 2:
      return "The template file denoted by the DATEMSK environment variable "
	     "cannot be opened.";
    case 3:
      return "Information about the template file cannot retrieved.";
    case 4:
      return "The template file is not a regular file.\n";
    case 5:
      return "An I/O error occurred while reading the template file.";
    case 6:
      return "Not enough memory available to execute the function.";
    case 7:
      return "The template file contains no matching template.";
    case 8:
      return "The input date is invalid, but would match a template "
	      "otherwise.";
    default:
      return "Unknown error code.";
    }
}

static char *datemsk;
static const char datemskstr[] =
  "%H:%M:%S %F\n"
  "%d-%m-%Y %T\n";

static void
do_prepare (int argc, char **argv)
{
  int fd = create_temp_file ("tst-chk1.", &datemsk);
  xwrite (fd, datemskstr, sizeof (datemskstr) - 1);

  setenv ("DATEMSK", datemsk, 1);
}
#define PREPARE do_prepare

static int
do_test (void)
{
  struct tm *tm;

  for (int i = 0; i < array_length (tests); ++i)
    {
      setenv ("TZ", tests[i].tz, 1);

      tm = getdate (tests[i].str);
      TEST_COMPARE (getdate_err, 0);
      if (getdate_err != 0)
	{
	  support_record_failure ();
	  printf ("%s\n", report_date_error ());
	}
      else
	{
	  TEST_COMPARE (tests[i].tm.tm_mon, tm->tm_mon);
	  TEST_COMPARE (tests[i].tm.tm_year, tm->tm_year);
	  TEST_COMPARE (tests[i].tm.tm_mday, tm->tm_mday);
	  TEST_COMPARE (tests[i].tm.tm_hour, tm->tm_hour);
	  TEST_COMPARE (tests[i].tm.tm_min, tm->tm_min);
	  TEST_COMPARE (tests[i].tm.tm_sec, tm->tm_sec);
	}

      struct tm tms;
      TEST_COMPARE (getdate_r (tests[i].str, &tms), 0);
      if (getdate_err == 0)
	{
	  TEST_COMPARE (tests[i].tm.tm_mon, tms.tm_mon);
	  TEST_COMPARE (tests[i].tm.tm_year, tms.tm_year);
	  TEST_COMPARE (tests[i].tm.tm_mday, tms.tm_mday);
	  TEST_COMPARE (tests[i].tm.tm_hour, tms.tm_hour);
	  TEST_COMPARE (tests[i].tm.tm_min, tms.tm_min);
	  TEST_COMPARE (tests[i].tm.tm_sec, tms.tm_sec);
	}
    }

  return 0;
}

#include <support/test-driver.c>
