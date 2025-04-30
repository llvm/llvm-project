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

#include <time.h>
#include <time-clockid.h>

/* Return the time now, and store it in *TIMER if not NULL.  */
time_t
time (time_t *timer)
{
  struct timespec ts;
  __clock_gettime (TIME_CLOCK_GETTIME_CLOCKID, &ts);

  if (timer)
    *timer = ts.tv_sec;
  return ts.tv_sec;
}
