/* utime -- Change access and modification times of file.  Linux version.
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

#include <utime.h>
#include <time.h>
#include <fcntl.h>

int
__utime64 (const char *file, const struct __utimbuf64 *times)
{
  struct __timespec64 ts64[2];

  if (times != NULL)
    {
      ts64[0].tv_sec = times->actime;
      ts64[0].tv_nsec = 0LL;
      ts64[1].tv_sec = times->modtime;
      ts64[1].tv_nsec = 0LL;
    }

  return __utimensat64_helper (AT_FDCWD, file, times ? ts64 : NULL, 0);
}

#if __TIMESIZE != 64
libc_hidden_def (__utime64)

int
__utime (const char *file, const struct utimbuf *times)
{
  struct __utimbuf64 utb64;

  if (times != NULL)
    {
      utb64.actime = (__time64_t) times->actime;
      utb64.modtime = (__time64_t) times->modtime;
    }

  return __utime64 (file, times ? &utb64 : NULL);
}
#endif
strong_alias (__utime, utime)
libc_hidden_def (utime)
