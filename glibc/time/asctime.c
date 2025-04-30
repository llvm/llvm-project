/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include "../locale/localeinfo.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <time.h>

/* This is defined in locale/C-time.c in the GNU libc.  */
extern const struct __locale_data _nl_C_LC_TIME attribute_hidden;
#define ab_day_name(DAY) (_nl_C_LC_TIME.values[_NL_ITEM_INDEX (ABDAY_1)+(DAY)].string)
#define ab_month_name(MON) (_nl_C_LC_TIME.values[_NL_ITEM_INDEX (ABMON_1)+(MON)].string)

static const char format[] = "%.3s %.3s%3d %.2d:%.2d:%.2d %d\n";
static char result[		 3+1+ 3+1+20+1+20+1+20+1+20+1+20+1 + 1];


static char *
asctime_internal (const struct tm *tp, char *buf, size_t buflen)
{
  if (tp == NULL)
    {
      __set_errno (EINVAL);
      return NULL;
    }

  /* We limit the size of the year which can be printed.  Using the %d
     format specifier used the addition of 1900 would overflow the
     number and a negative vaue is printed.  For some architectures we
     could in theory use %ld or an evern larger integer format but
     this would mean the output needs more space.  This would not be a
     problem if the 'asctime_r' interface would be defined sanely and
     a buffer size would be passed.  */
  if (__glibc_unlikely (tp->tm_year > INT_MAX - 1900))
    {
    eoverflow:
      __set_errno (EOVERFLOW);
      return NULL;
    }

  int n = __snprintf (buf, buflen, format,
		      (tp->tm_wday < 0 || tp->tm_wday >= 7
		       ? "???" : ab_day_name (tp->tm_wday)),
		      (tp->tm_mon < 0 || tp->tm_mon >= 12
		       ? "???" : ab_month_name (tp->tm_mon)),
		      tp->tm_mday, tp->tm_hour, tp->tm_min,
		      tp->tm_sec, 1900 + tp->tm_year);
  if (n < 0)
    return NULL;
  if (n >= buflen)
    goto eoverflow;

  return buf;
}


/* Like asctime, but write result to the user supplied buffer.  The
   buffer is only guaranteed to be 26 bytes in length.  */
char *
__asctime_r (const struct tm *tp, char *buf)
{
  return asctime_internal (tp, buf, 26);
}
weak_alias (__asctime_r, asctime_r)


/* Returns a string of the form "Day Mon dd hh:mm:ss yyyy\n"
   which is the representation of TP in that form.  */
char *
asctime (const struct tm *tp)
{
  return asctime_internal (tp, result, sizeof (result));
}
libc_hidden_def (asctime)
