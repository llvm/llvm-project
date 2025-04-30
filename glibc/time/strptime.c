/* Convert a string representation of time to a time value.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

/* XXX This version of the implementation is not really complete.
   Some of the fields cannot add information alone.  But if seeing
   some of them in the same format (such as year, week and weekday)
   this is enough information for determining the date.  */

#include <time.h>
#include <locale/localeinfo.h>


char *
strptime (const char *buf, const char *format, struct tm *tm)
{
  return __strptime_internal (buf, format, tm, NULL, _NL_CURRENT_LOCALE);
}
libc_hidden_def (strptime)
