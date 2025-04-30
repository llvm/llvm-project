/* dirfd -- Return the file descriptor used by a DIR stream.  Hurd version.
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

#include <dirent.h>
#include <dirstream.h>
#include <hurd/fd.h>
#include <errno.h>

int
__dirfd (DIR *dirp)
{
  int fd;

  HURD_CRITICAL_BEGIN;
  __mutex_lock (&_hurd_dtable_lock);
  for (fd = 0; fd < _hurd_dtablesize; ++fd)
    if (_hurd_dtable[fd] == dirp->__fd)
      break;
  if (fd == _hurd_dtablesize)
    {
      errno = EINVAL;
      fd = -1;
    }
  __mutex_unlock (&_hurd_dtable_lock);
  HURD_CRITICAL_END;

  return fd;
}

weak_alias (__dirfd, dirfd)
libc_hidden_def (dirfd)
