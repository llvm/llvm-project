/* utime -- Change access and modification times of file.  Posix version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <errno.h>
#include <utime.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>


/* Set the access and modification times of FILE to those given in TIMES.
   If TIMES is NULL, set them to the current time.  */
int
utime (const char *file, const struct utimbuf *times)
{
  struct timeval timevals[2];
  struct timeval *tvp;

  if (times != NULL)
    {
      timevals[0].tv_sec = (time_t) times->actime;
      timevals[0].tv_usec = 0L;
      timevals[1].tv_sec = (time_t) times->modtime;
      timevals[1].tv_usec = 0L;
      tvp = timevals;
    }
  else
    tvp = NULL;

  return __utimes (file, tvp);
}
libc_hidden_def (utime)
