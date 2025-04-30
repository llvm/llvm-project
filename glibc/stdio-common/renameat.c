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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>


/* Rename the file OLD relative to OLDFD to NEW relative to NEWFD.  */
int
__renameat (int oldfd, const char *old, int newfd, const char *new)
{
  if ((oldfd < 0 && oldfd != AT_FDCWD) || (newfd < 0 && newfd != AT_FDCWD))
    {
      __set_errno (EBADF);
      return -1;
    }

  if (old == NULL || new == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  __set_errno (ENOSYS);
  return -1;
}

libc_hidden_def (__renameat)
weak_alias (__renameat, renameat)
stub_warning (renameat)
