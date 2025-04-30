/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#include <sys/statvfs.h>
#include <sys/statfs.h>
#include <internal_statvfs.h>
#include <time.h>
#include <kernel_stat.h>

#if !STATFS_IS_STATFS64
int
__statvfs (const char *file, struct statvfs *buf)
{
  struct statfs fsbuf;

  /* Get as much information as possible from the system.  */
  if (__statfs (file, &fsbuf) < 0)
    return -1;

  /* Convert the result.  */
  __internal_statvfs (buf, &fsbuf);

  /* We signal success if the statfs call succeeded.  */
  return 0;
}
weak_alias (__statvfs, statvfs)
libc_hidden_weak (statvfs)
#endif
