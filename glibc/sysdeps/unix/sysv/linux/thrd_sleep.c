/* C11 threads thread sleep implementation - Linux variant.
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

#include <time.h>
#include <sysdep-cancel.h>

#include "thrd_priv.h"

int
__thrd_sleep64 (const struct __timespec64 *time_point,
                struct __timespec64 *remaining)
{
  int ret = __clock_nanosleep_time64 (CLOCK_REALTIME, 0, time_point,
                                      remaining);
  /* C11 states thrd_sleep function returns -1 if it has been interrupted
     by a signal, or a negative value if it fails.  */
  switch (ret)
  {
     case 0:      return 0;
     case EINTR:  return -1;
     default:     return -2;
  }
}

#if __TIMESIZE != 64
libc_hidden_def (__thrd_sleep64)

int
__thrd_sleep (const struct timespec *time_point, struct timespec *remaining)
{
  const struct __timespec64 treq64 = valid_timespec_to_timespec64 (*time_point);
  struct __timespec64 trem64;

  int ret = __thrd_sleep64 (&treq64, remaining != NULL ? &trem64 : NULL);
  if (ret == -1 && remaining != NULL)
    *remaining = valid_timespec64_to_timespec (trem64);

  return ret;
}
#endif
weak_alias (__thrd_sleep, thrd_sleep)
