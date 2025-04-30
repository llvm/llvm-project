/* futimes -- change access and modification times of open file.  Hurd version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <sys/time.h>
#include <errno.h>
#include <stddef.h>
#include <hurd.h>
#include <hurd/fd.h>

#include "utime-helper.c"

/* Change the access time of FD to TVP[0] and
   the modification time of FD to TVP[1].  */
int
__futimes (int fd, const struct timeval tvp[2])
{
  struct timespec atime, mtime;
  error_t err;

  utime_ts_from_tval (tvp, &atime, &mtime);

  err = HURD_DPORT_USE (fd, __file_utimens (port, atime, mtime));

  if (err == EMIG_BAD_ID || err == EOPNOTSUPP)
    {
      time_value_t atim, mtim;

      utime_tvalue_from_tval (tvp, &atim, &mtim);

      err = HURD_DPORT_USE (fd, __file_utimes (port, atim, mtim));
    }

  return err ? __hurd_dfail (fd, err) : 0;
}
weak_alias (__futimes, futimes)
