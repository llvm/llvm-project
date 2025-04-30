/* settimeofday -- set system time - Linux version supporting 64 bit time.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
__settimeofday64 (const struct __timeval64 *tv, const struct timezone *tz)
{
  /* Backwards compatibility for setting the UTC offset.  */
  if (__glibc_unlikely (tz != 0))
    {
      if (tv != 0)
	{
	  __set_errno (EINVAL);
	  return -1;
	}
      return __settimezone (tz);
    }

  struct __timespec64 ts = timeval64_to_timespec64 (*tv);
  return __clock_settime64 (CLOCK_REALTIME, &ts);
}

#if __TIMESIZE != 64
libc_hidden_def (__settimeofday64)

int
__settimeofday (const struct timeval *tv, const struct timezone *tz)
{
  if (__glibc_unlikely (tv == NULL))
    return __settimeofday64 (NULL, tz);
  else
    {
      struct __timeval64 tv64 = valid_timeval_to_timeval64 (*tv);
      return __settimeofday64 (&tv64, tz);
    }
}
#endif
weak_alias (__settimeofday, settimeofday);
