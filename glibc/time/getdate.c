/* Convert a string representation of time to a time value.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Mark Kettenis <kettenis@phys.uva.nl>, 1997.

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

#include <limits.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <ctype.h>
#include <alloca.h>

#define TM_YEAR_BASE 1900


/* Prototypes for local functions.  */
static int first_wday (int year, int mon, int wday);
static int check_mday (int year, int mon, int mday);


/* Set to one of the following values to indicate an error.
     1  the DATEMSK environment variable is null or undefined,
     2  the template file cannot be opened for reading,
     3  failed to get file status information,
     4  the template file is not a regular file,
     5  an error is encountered while reading the template file,
     6  memory allication failed (not enough memory available),
     7  there is no line in the template that matches the input,
     8  invalid input specification Example: February 31 or a time is
	specified that can not be represented in a time_t (representing
	the time in seconds since 00:00:00 UTC, January 1, 1970) */
int getdate_err;


/* Returns the first weekday WDAY of month MON in the year YEAR.  */
static int
first_wday (int year, int mon, int wday)
{
  struct tm tm;

  if (wday == INT_MIN)
    return 1;

  memset (&tm, 0, sizeof (struct tm));
  tm.tm_year = year;
  tm.tm_mon = mon;
  tm.tm_mday = 1;
  mktime (&tm);

  return (1 + (wday - tm.tm_wday + 7) % 7);
}


/* Returns 1 if MDAY is a valid day of the month in month MON of year
   YEAR, and 0 if it is not.  */
static int
check_mday (int year, int mon, int mday)
{
  switch (mon)
    {
    case 0:
    case 2:
    case 4:
    case 6:
    case 7:
    case 9:
    case 11:
      if (mday >= 1 && mday <= 31)
	return 1;
      break;
    case 3:
    case 5:
    case 8:
    case 10:
      if (mday >= 1 && mday <= 30)
	return 1;
      break;
    case 1:
      if (mday >= 1 && mday <= (__isleap (year) ? 29 : 28))
	return 1;
      break;
    }

  return 0;
}


int
__getdate_r (const char *string, struct tm *tp)
{
  FILE *fp;
  char *line;
  size_t len;
  char *datemsk;
  char *result = NULL;
  __time64_t timer;
  struct tm tm;
  struct __stat64_t64 st;
  bool mday_ok = false;

  datemsk = getenv ("DATEMSK");
  if (datemsk == NULL || *datemsk == '\0')
    return 1;

  if (__stat64_time64 (datemsk, &st) < 0)
    return 3;

  if (!S_ISREG (st.st_mode))
    return 4;

  if (__access (datemsk, R_OK) < 0)
    return 2;

  /* Open the template file.  */
  fp = fopen (datemsk, "rce");
  if (fp == NULL)
    return 2;

  /* No threads reading this stream.  */
  __fsetlocking (fp, FSETLOCKING_BYCALLER);

  /* Skip leading whitespace.  */
  while (isspace (*string))
    string++;

  size_t inlen, oldlen;

  oldlen = inlen = strlen (string);

  /* Skip trailing whitespace.  */
  while (inlen > 0 && isspace (string[inlen - 1]))
    inlen--;

  char *instr = NULL;

  if (inlen < oldlen)
    {
      bool using_malloc = false;

      if (__libc_use_alloca (inlen + 1))
	instr = alloca (inlen + 1);
      else
	{
	  instr = malloc (inlen + 1);
	  if (instr == NULL)
	    {
	      fclose (fp);
	      return 6;
	    }
	  using_malloc = true;
	}
      memcpy (instr, string, inlen);
      instr[inlen] = '\0';
      string = instr;

      if (!using_malloc)
	instr = NULL;
    }

  line = NULL;
  len = 0;
  do
    {
      ssize_t n;

      n = __getline (&line, &len, fp);
      if (n < 0)
	break;
      if (line[n - 1] == '\n')
	line[n - 1] = '\0';

      /* Do the conversion.  */
      tp->tm_year = tp->tm_mon = tp->tm_mday = tp->tm_wday = INT_MIN;
      tp->tm_hour = tp->tm_sec = tp->tm_min = INT_MIN;
      tp->tm_isdst = -1;
      tp->tm_gmtoff = 0;
      tp->tm_zone = NULL;
      result = strptime (string, line, tp);
      if (result && *result == '\0')
	break;
    }
  while (!__feof_unlocked (fp));

  free (instr);

  /* Free the buffer.  */
  free (line);

  /* Check for errors. */
  if (__ferror_unlocked (fp))
    {
      fclose (fp);
      return 5;
    }

  /* Close template file.  */
  fclose (fp);

  if (result == NULL || *result != '\0')
    return 7;

  /* Get current time.  */
  timer = time64_now ();
  __localtime64_r (&timer, &tm);

  /* If only the weekday is given, today is assumed if the given day
     is equal to the current day and next week if it is less.  */
  if (tp->tm_wday >= 0 && tp->tm_wday <= 6 && tp->tm_year == INT_MIN
      && tp->tm_mon == INT_MIN && tp->tm_mday == INT_MIN)
    {
      tp->tm_year = tm.tm_year;
      tp->tm_mon = tm.tm_mon;
      tp->tm_mday = tm.tm_mday + (tp->tm_wday - tm.tm_wday + 7) % 7;
      mday_ok = true;
    }

  /* If only the month is given, the current month is assumed if the
     given month is equal to the current month and next year if it is
     less and no year is given (the first day of month is assumed if
     no day is given.  */
  if (tp->tm_mon >= 0 && tp->tm_mon <= 11 && tp->tm_mday == INT_MIN)
    {
      if (tp->tm_year == INT_MIN)
	tp->tm_year = tm.tm_year + (((tp->tm_mon - tm.tm_mon) < 0) ? 1 : 0);
      tp->tm_mday = first_wday (tp->tm_year, tp->tm_mon, tp->tm_wday);
      mday_ok = true;
    }

  /* If no hour, minute and second are given the current hour, minute
     and second are assumed.  */
  if (tp->tm_hour == INT_MIN && tp->tm_min == INT_MIN && tp->tm_sec == INT_MIN)
    {
      tp->tm_hour = tm.tm_hour;
      tp->tm_min = tm.tm_min;
      tp->tm_sec = tm.tm_sec;
    }

  /* Fill in the gaps.  */
  if (tp->tm_hour == INT_MIN)
    tp->tm_hour = 0;
  if (tp->tm_min == INT_MIN)
    tp->tm_min = 0;
  if (tp->tm_sec == INT_MIN)
    tp->tm_sec = 0;

  /* If no date is given, today is assumed if the given hour is
     greater than the current hour and tomorrow is assumed if
     it is less.  */
  if (tp->tm_hour >= 0 && tp->tm_hour <= 23
      && tp->tm_mon == INT_MIN
      && tp->tm_mday == INT_MIN && tp->tm_wday == INT_MIN)
    {
      tp->tm_mon = tm.tm_mon;
      tp->tm_mday = tm.tm_mday + ((tp->tm_hour - tm.tm_hour) < 0 ? 1 : 0);
      mday_ok = 1;
    }

  /* More fillers.  */
  if (tp->tm_year == INT_MIN)
    tp->tm_year = tm.tm_year;
  if (tp->tm_mon == INT_MIN)
    tp->tm_mon = tm.tm_mon;

  /* Check if the day of month is within range, and if the time can be
     represented in a time_t.  We make use of the fact that the mktime
     call normalizes the struct tm.  */
  if ((!mday_ok && !check_mday (TM_YEAR_BASE + tp->tm_year, tp->tm_mon,
				tp->tm_mday))
      || __mktime64 (tp) == (time_t) -1)
    return 8;

  return 0;
}
weak_alias (__getdate_r, getdate_r)
libc_hidden_def (__getdate_r)

struct tm *
getdate (const char *string)
{
  /* Buffer returned by getdate.  */
  static struct tm tmbuf;
  int errval = __getdate_r (string, &tmbuf);

  if (errval != 0)
    {
      getdate_err = errval;
      return NULL;
    }

  return &tmbuf;
}
