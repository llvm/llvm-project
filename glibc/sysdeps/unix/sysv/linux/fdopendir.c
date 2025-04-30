/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <sys/stat.h>

#include <not-cancel.h>


DIR *
__fdopendir (int fd)
{
  struct __stat64_t64 statbuf;

  if (__glibc_unlikely (__fstat64_time64 (fd, &statbuf) < 0))
    return NULL;
  if (__glibc_unlikely (! S_ISDIR (statbuf.st_mode)))
    {
      __set_errno (ENOTDIR);
      return NULL;
    }

  /* Make sure the descriptor allows for reading.  */
  int flags = __fcntl64_nocancel (fd, F_GETFL);
  if (__glibc_unlikely (flags == -1))
    return NULL;
  if (__glibc_unlikely ((flags & O_ACCMODE) == O_WRONLY))
    {
      __set_errno (EINVAL);
      return NULL;
    }

  return __alloc_dir (fd, false, flags, &statbuf);
}
weak_alias (__fdopendir, fdopendir)
