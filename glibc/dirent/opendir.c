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
#include <stddef.h>
#include <dirent.h>
#include <fcntl.h>


DIR *
__opendirat (int dfd, const char *name)
{
  __set_errno (ENOSYS);
  return NULL;
}


/* Open a directory stream on NAME.  */
DIR *
__opendir (const char *name)
{
  return __opendirat (AT_FDCWD, name);
}
weak_alias (__opendir, opendir)

stub_warning (opendir)
