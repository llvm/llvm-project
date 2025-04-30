/* Test for strptime.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


static const struct
{
  const char *locale;
  const char *input;
  const char *format;
  int wday;
  int yday;
  int mon;
  int mday;
} day_tests[] =
{
  { "C", "2000-01-01", "%Y-%m-%d", 6, 0, 0, 1 },
  { "C", "03/03/00", "%D", 5, 62, 2, 3 },
  { "C", "9/9/99", "%x", 4, 251, 8, 9 },
  { "C", "19990502123412", "%Y%m%d%H%M%S", 0, 121, 4, 2 },
  { "C", "2001 20 Mon", "%Y %U %a", 1, 140, 4, 21 },
  { "C", "2001 21 Mon", "%Y %W %a", 1, 140, 4, 21 },
  { "C", "2001 21 Mon", "%2000Y %W %a", 1, 140, 4, 21 },
  { "C", "2001 21 Mon", "%^Y %W %a", 1, 140, 4, 21 },
  { "C", "2001 EST 21 Mon", "%Y %Z %W %a", 1, 140, 4, 21 },
  { "C", "2012 00 Sun", "%Y %W %a", 0, 0, 0, 1 },
  { "ja_JP.EUC-JP", "2000-01-01 08:12:21 AM", "%Y-%m-%d %I:%M:%S %p",
    6, 0, 0, 1 },
  { "en_US.ISO-8859-1", "2000-01-01 08:12:21 PM", "%Y-%m-%d %I:%M:%S %p",
    6, 0, 0, 1 },
  { "ja_JP.EUC-JP", "2001 20 \xb7\xee", "%Y %U %a", 1, 140, 4, 21 },
  { "ja_JP.EUC-JP", "2001 21 \xb7\xee", "%Y %W %a", 1, 140, 4, 21 },
  /* Most of the languages do not need the declension of the month names
     and do not distinguish between %B and %OB.  */
  { "en_US.ISO-8859-1", "November 17, 2017", "%B %e, %Y", 5, 320, 10, 17 },
  { "de_DE.ISO-8859-1", "18. Nov 2017", "%d. %b %Y", 6, 321, 10, 18 },
  { "fr_FR.UTF-8", "19 novembre 2017", "%d %OB %Y", 0, 322, 10, 19 },
  { "es_ES.UTF-8", "20 de nov de 2017", "%d de %Ob de %Y", 1, 323, 10, 20 },
  /* Some languages do need the declension of the month names.  */
  { "pl_PL.UTF-8", "21 lis 2017", "%d %b %Y", 2, 324, 10, 21 },
  { "pl_PL.UTF-8", "22 LIS 2017", "%d %B %Y", 3, 325, 10, 22 },
  { "pl_PL.UTF-8", "23 listopada 2017", "%d %B %Y", 4, 326, 10, 23 },
  /* The nominative case is incorrect here but it is parseable.  */
  { "pl_PL.UTF-8", "24 listopad 2017", "%d %OB %Y", 5, 327, 10, 24 },
  { "pl_PL.UTF-8", "25 lis 2017", "%d %Ob %Y", 6, 328, 10, 25 },
  /* ноя - pronounce: 'noya' - "Nov" (abbreviated "November") in Russian.  */
  { "ru_RU.UTF-8", "26 ноя 2017", "%d %b %Y", 0, 329, 10, 26 },
  /* мая - pronounce: 'maya' - "of May" (the genitive case, both full and
     abbreviated) in Russian.  */
  { "ru_RU.UTF-8", "19 мая 2018", "%d %b %Y", 6, 138, 4, 19 },
  /* май - pronounce: 'may' - "May" (the nominative case, both full and
     abbreviated) in Russian.
     The nominative case is incorrect here but it is parseable.  */
  { "ru_RU.UTF-8", "20 май 2018", "%d %Ob %Y", 0, 139, 4, 20 },
};


static const struct
{
  const char *input;
  const char *format;
  const char *output;
  int wday;
  int yday;
} tm_tests [] =
{
  {"17410105012000", "%H%M%S%d%m%Y", "2000-01-05 17:41:01", 3, 4}
};



static int
test_tm (void)
{
  struct tm tm;
  size_t i;
  int result = 0;
  char buf[100];

  for (i = 0; i < sizeof (tm_tests) / sizeof (tm_tests[0]); ++i)
    {
      memset (&tm, '\0', sizeof (tm));

      char *ret = strptime (tm_tests[i].input, tm_tests[i].format, &tm);
      if (ret == NULL)
	{
	  printf ("strptime returned NULL for `%s'\n", tm_tests[i].input);
	  result = 1;
	  continue;
	}
      else if (*ret != '\0')
	{
	  printf ("not all of `%s' read\n", tm_tests[i].input);
	  result = 1;
	}
      strftime (buf, sizeof (buf), "%F %T", &tm);
      printf ("strptime (\"%s\", \"%s\", ...)\n"
	      "\tshould be: %s, wday = %d, yday = %3d\n"
	      "\t       is: %s, wday = %d, yday = %3d\n",
	      tm_tests[i].input, tm_tests[i].format,
	      tm_tests[i].output,
	      tm_tests[i].wday, tm_tests[i].yday,
	      buf, tm.tm_wday, tm.tm_yday);

      if (strcmp (buf, tm_tests[i].output) != 0)
	{
	  printf ("Time and date are not correct.\n");
	  result = 1;
	}
      if (tm.tm_wday != tm_tests[i].wday)
	{
	  printf ("weekday for `%s' incorrect: %d instead of %d\n",
		  tm_tests[i].input, tm.tm_wday, tm_tests[i].wday);
	  result = 1;
	}
      if (tm.tm_yday != tm_tests[i].yday)
	{
	  printf ("yearday for `%s' incorrect: %d instead of %d\n",
		  tm_tests[i].input, tm.tm_yday, tm_tests[i].yday);
	  result = 1;
	}
    }

  return result;
}


static int
do_test (void)
{
  struct tm tm;
  size_t i;
  int result = 0;

  for (i = 0; i < sizeof (day_tests) / sizeof (day_tests[0]); ++i)
    {
      memset (&tm, '\0', sizeof (tm));

      if (setlocale (LC_ALL, day_tests[i].locale) == NULL)
	{
	  printf ("cannot set locale %s: %m\n", day_tests[i].locale);
	  exit (EXIT_FAILURE);
	}

      char *ret = strptime (day_tests[i].input, day_tests[i].format, &tm);
      if (ret == NULL)
	{
	  printf ("strptime returned NULL for `%s'\n", day_tests[i].input);
	  result = 1;
	  continue;
	}
      else if (*ret != '\0')
	{
	  printf ("not all of `%s' read\n", day_tests[i].input);
	  result = 1;
	}

      printf ("strptime (\"%s\", \"%s\", ...)\n"
	      "\tshould be: wday = %d, yday = %3d, mon = %2d, mday = %2d\n"
	      "\t       is: wday = %d, yday = %3d, mon = %2d, mday = %2d\n",
	      day_tests[i].input, day_tests[i].format,
	      day_tests[i].wday, day_tests[i].yday,
	      day_tests[i].mon, day_tests[i].mday,
	      tm.tm_wday, tm.tm_yday, tm.tm_mon, tm.tm_mday);

      if (tm.tm_wday != day_tests[i].wday)
	{
	  printf ("weekday for `%s' incorrect: %d instead of %d\n",
		  day_tests[i].input, tm.tm_wday, day_tests[i].wday);
	  result = 1;
	}
      if (tm.tm_yday != day_tests[i].yday)
	{
	  printf ("yearday for `%s' incorrect: %d instead of %d\n",
		  day_tests[i].input, tm.tm_yday, day_tests[i].yday);
	  result = 1;
	}
      if (tm.tm_mon != day_tests[i].mon)
	{
	  printf ("month for `%s' incorrect: %d instead of %d\n",
		  day_tests[i].input, tm.tm_mon, day_tests[i].mon);
	  result = 1;
	}
      if (tm.tm_mday != day_tests[i].mday)
	{
	  printf ("monthday for `%s' incorrect: %d instead of %d\n",
		  day_tests[i].input, tm.tm_mday, day_tests[i].mday);
	  result = 1;
	}
    }

  setlocale (LC_ALL, "C");

  result |= test_tm ();

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
