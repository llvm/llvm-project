/* Get resolution of a time base.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <time.h>

/* Set TS to resolution of time base BASE.  */
int
__timespec_getres64 (struct __timespec64 *ts, int base)
{
  if (base == TIME_UTC)
    {
      __clock_getres64 (CLOCK_REALTIME, ts);
      return base;
    }
  return 0;
}

#if __TIMESIZE != 64
libc_hidden_def (__timespec_getres64)

int
__timespec_getres (struct timespec *ts, int base)
{
  int ret;
  struct __timespec64 tp64;

  ret = __timespec_getres64 (&tp64, base);

  if (ret == TIME_UTC && ts != NULL)
    *ts = valid_timespec64_to_timespec (tp64);

  return ret;
}
#endif
strong_alias (__timespec_getres, timespec_getres);
