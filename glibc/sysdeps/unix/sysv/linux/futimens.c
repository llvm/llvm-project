/* Change access and modification times of open file.  Linux version.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <sysdep.h>


/* Change the access time of the file associated with FD to TSP[0] and
   the modification time of FILE to TSP[1].

   Starting with 2.6.22 the Linux kernel has the utimensat syscall which
   can be used to implement futimens.  */
int
__futimens64 (int fd, const struct __timespec64 tsp64[2])
{
  if (fd < 0)
    return INLINE_SYSCALL_ERROR_RETURN_VALUE (EBADF);

  return __utimensat64_helper (fd, NULL, &tsp64[0], 0);
}

#if __TIMESIZE != 64
libc_hidden_def (__futimens64);

int
__futimens (int fd, const struct timespec tsp[2])
{
  struct __timespec64 tsp64[2];
  if (tsp)
    {
      tsp64[0] = valid_timespec_to_timespec64 (tsp[0]);
      tsp64[1] = valid_timespec_to_timespec64 (tsp[1]);
    }

  return __futimens64 (fd, tsp ? &tsp64[0] : NULL);
}
#endif
weak_alias (__futimens, futimens)
