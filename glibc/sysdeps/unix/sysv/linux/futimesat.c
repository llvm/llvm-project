/* futimesat -- Change access and modification times of file.  Linux version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

/* Change the access time of FILE relative to FD to TVP[0] and
   the modification time of FILE to TVP[1].  */
int
__futimesat64 (int fd, const char *file, const struct __timeval64 tvp64[2])
{
  struct __timespec64 ts64[2];
  if (tvp64 != NULL)
    {
      ts64[0] = timeval64_to_timespec64 (tvp64[0]);
      ts64[1] = timeval64_to_timespec64 (tvp64[1]);
    }

  return __utimensat64_helper (fd, file, tvp64 ? &ts64[0] : NULL, 0);
}

#if __TIMESIZE != 64
libc_hidden_def (__futimesat64)

int
__futimesat (int fd, const char *file, const struct timeval tvp[2])
{
  struct __timeval64 tv64[2];

  if (tvp != NULL)
    {
      tv64[0] = valid_timeval_to_timeval64 (tvp[0]);
      tv64[1] = valid_timeval_to_timeval64 (tvp[1]);
    }

  return __futimesat64 (fd, file, tvp ? &tv64[0] : NULL);
}
#endif
weak_alias (__futimesat, futimesat)
