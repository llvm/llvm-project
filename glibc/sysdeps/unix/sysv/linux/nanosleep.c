/* High-resolution sleep.
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
#include <errno.h>

int
__nanosleep64 (const struct __timespec64 *req, struct __timespec64 *rem)
{
  int ret = __clock_nanosleep_time64 (CLOCK_REALTIME, 0, req, rem);
  if (ret != 0)
    {
      __set_errno (ret);
      return -1;
    }
  return 0;
}
#if __TIMESIZE != 64
libc_hidden_def (__nanosleep64)

int
__nanosleep (const struct timespec *req, struct timespec *rem)
{
  struct __timespec64 treq64, trem64;

  treq64 = valid_timespec_to_timespec64 (*req);
  int ret = __nanosleep64 (&treq64, rem != NULL ? &trem64 : NULL);

  if (ret != 0 && errno == EINTR && rem != NULL)
    *rem = valid_timespec64_to_timespec (trem64);
  return ret;
}
#endif
libc_hidden_def (__nanosleep)
weak_alias (__nanosleep, nanosleep)
