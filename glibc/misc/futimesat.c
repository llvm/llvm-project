/* futimesat -- Change access and modification times of file.  Stub version.
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
#include <fcntl.h>
#include <stddef.h>
#include <sys/time.h>


/* Change the access time of FILE relative to FD to TVP[0] and
   the modification time of FILE to TVP[1].  */
int
futimesat (int fd, const char *file, const struct timeval tvp[2])
{
  if (fd < 0
      && (file == NULL
          || (fd != AT_FDCWD && file[0] != '/')))
    {
      __set_errno (EBADF);
      return -1;
    }

  __set_errno (ENOSYS);
  return -1;
}

stub_warning (futimesat)
