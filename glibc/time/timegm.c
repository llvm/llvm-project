/* Convert UTC calendar time to simple time.  Like mktime but assumes UTC.

   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef _LIBC
# include <libc-config.h>
#endif

#include <time.h>
#include <errno.h>

#include "mktime-internal.h"

__time64_t
__timegm64 (struct tm *tmp)
{
  static mktime_offset_t gmtime_offset;
  tmp->tm_isdst = 0;
  return __mktime_internal (tmp, __gmtime64_r, &gmtime_offset);
}

#if defined _LIBC && __TIMESIZE != 64

libc_hidden_def (__timegm64)

time_t
timegm (struct tm *tmp)
{
  struct tm tm = *tmp;
  __time64_t t = __timegm64 (&tm);
  if (in_time_t_range (t))
    {
      *tmp = tm;
      return t;
    }
  else
    {
      __set_errno (EOVERFLOW);
      return -1;
    }
}

#endif
