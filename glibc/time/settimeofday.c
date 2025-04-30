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

#include <errno.h>
#include <time.h>
#include <sys/time.h>

/* Set the current time of day and timezone information.
   This call is restricted to the super-user.  */
int
__settimeofday (const struct timeval *tv, const struct timezone *tz)
{
  if (__glibc_unlikely (tz != 0))
    {
      if (tv != 0)
	{
	  __set_errno (EINVAL);
	  return -1;
	}
      return __settimezone (tz);
    }

  struct timespec ts;
  TIMEVAL_TO_TIMESPEC (tv, &ts);
  return __clock_settime (CLOCK_REALTIME, &ts);
}

#ifdef VERSION_settimeofday
weak_alias (__settimeofday, __settimeofday_w);
default_symbol_version (__settimeofday_w, settimeofday, VERSION_settimeofday);
#else
weak_alias (__settimeofday, settimeofday);
#endif
