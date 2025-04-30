/* utimes -- Change access and modification times of file.  Linux version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#include <fcntl.h>

int
__utimes64 (const char *file, const struct __timeval64 tvp[2])
{
  struct __timespec64 ts64[2];

  if (tvp != NULL)
    {
      ts64[0] = timeval64_to_timespec64 (tvp[0]);
      ts64[1] = timeval64_to_timespec64 (tvp[1]);
    }

  return __utimensat64_helper (AT_FDCWD, file, tvp ? ts64 : NULL, 0);
}

#if __TIMESIZE != 64
libc_hidden_def (__utimes64)

int
__utimes (const char *file, const struct timeval tvp[2])
{
  struct __timeval64 tv64[2];

  if (tvp != NULL)
    {
      tv64[0] = valid_timeval_to_timeval64 (tvp[0]);
      tv64[1] = valid_timeval_to_timeval64 (tvp[1]);
    }

  return __utimes64 (file, tvp ? tv64 : NULL);
}
#endif
weak_alias (__utimes, utimes)
