/* Rename a file using relative source and destination names.  Hurd version.
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

#include <stdio.h>
#include <hurd.h>
#include <hurd/fd.h>

/* Rename the file OLD relative to OLDFD to NEW relative to NEWFD.  */
int
__renameat (int oldfd, const char *old, int newfd, const char *new)
{
  return __renameat2 (oldfd, old, newfd, new, 0);
}
libc_hidden_def (__renameat)
weak_alias (__renameat, renameat)
