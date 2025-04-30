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

#include <string.h>
#include <time.h>
#include <sys/time.h>

/* Get the current time of day, putting it into *TV.
   If *TZ is not NULL, clear it.
   Returns 0 on success, -1 on errors.  */
int
___gettimeofday (struct timeval *restrict tv, void *restrict tz)
{
  if (__glibc_unlikely (tz != 0))
    memset (tz, 0, sizeof (struct timezone));

  struct timespec ts;
  if (__clock_gettime (CLOCK_REALTIME, &ts))
    return -1;

  TIMESPEC_TO_TIMEVAL (tv, &ts);
  return 0;
}
/* Define to override default symbol version.  */
#ifndef SET_VERSION
strong_alias (___gettimeofday, __gettimeofday)
weak_alias (___gettimeofday, gettimeofday)
#endif
