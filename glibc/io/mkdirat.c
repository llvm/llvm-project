/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sys/stat.h>
#include <sys/types.h>


/* Create a directory named PATH relative to FD with protections MODE.  */
int
mkdirat (int fd, const char *path, mode_t mode)
{
  if (path == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  if (fd != AT_FDCWD && path[0] != '/')
    {
      /* Check FD is associated with a directory.  */
      struct stat64 st;
      if (__fstat64 (fd, &st) != 0)
	return -1;

      if (!S_ISDIR (st.st_mode))
	{
	  __set_errno (ENOTDIR);
	  return -1;
	}
    }

  __set_errno (ENOSYS);
  return -1;
}
stub_warning (mkdirat)
