/* Deprecated return date and time.  Linux version.
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

#include <features.h>
#include <sys/timeb.h>
#include <time.h>
#include <errno.h>

int
__ftime64 (struct __timeb64 *timebuf)
{
  struct __timespec64 ts;
  __clock_gettime64 (CLOCK_REALTIME, &ts);

  timebuf->time = ts.tv_sec;
  timebuf->millitm = ts.tv_nsec / 1000000;
  timebuf->timezone = 0;
  timebuf->dstflag = 0;
  return 0;
}
#if __TIMESIZE != 64
libc_hidden_def (__ftime64)

int
ftime (struct timeb *timebuf)
{
  struct __timeb64 tb64;
  __ftime64 (&tb64);
  if (! in_time_t_range (tb64.time))
    {
      __set_errno (EOVERFLOW);
      return -1;
    }
  timebuf->time = tb64.time;
  timebuf->millitm = tb64.millitm;
  timebuf->timezone = tb64.timezone;
  timebuf->dstflag = tb64.dstflag;
  return 0;
}
#endif
