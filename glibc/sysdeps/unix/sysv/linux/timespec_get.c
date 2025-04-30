/* timespec_get -- get system time - Linux version supporting 64 bit time.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <errno.h>

/* Set TS to calendar time based in time base BASE.  */
int
__timespec_get64 (struct __timespec64 *ts, int base)
{
  if (base == TIME_UTC)
    {
      __clock_gettime64 (CLOCK_REALTIME, ts);
      return base;
    }
  return 0;
}

#if __TIMESIZE != 64
libc_hidden_def (__timespec_get64)

int
__timespec_get (struct timespec *ts, int base)
{
  int ret;
  struct __timespec64 tp64;

  ret = __timespec_get64 (&tp64, base);

  if (ret == TIME_UTC)
    {
      if (! in_time_t_range (tp64.tv_sec))
        {
          __set_errno (EOVERFLOW);
          return 0;
        }

      *ts = valid_timespec64_to_timespec (tp64);
    }

  return ret;
}
#endif
strong_alias (__timespec_get, timespec_get);
