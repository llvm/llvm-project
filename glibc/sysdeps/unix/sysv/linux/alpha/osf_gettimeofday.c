/* gettimeofday -- Get the current time of day.  Linux/Alpha/tv32 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

#include <time.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

/* Get the current time of day and timezone information putting it
   into *TV and *TZ.  */

int
attribute_compat_text_section
__gettimeofday_tv32 (struct __timeval32 *restrict tv32, void *restrict tz)
{
  if (__glibc_unlikely (tz != 0))
    memset (tz, 0, sizeof (struct timezone));

  struct __timespec64 ts;
  __clock_gettime64 (CLOCK_REALTIME, &ts);

  *tv32 = valid_timespec_to_timeval32 (ts);
  return 0;
}

compat_symbol (libc, __gettimeofday_tv32, __gettimeofday, GLIBC_2_0);
strong_alias (__gettimeofday_tv32, __gettimeofday_tv32_1);
compat_symbol (libc, __gettimeofday_tv32_1, gettimeofday, GLIBC_2_0);
#endif
