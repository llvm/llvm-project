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
#include <unistd.h>
#include <dirstream.h>

/* Seek to position POS in DIRP.  */
/* XXX should be __seekdir ? */
void
seekdir (DIR *dirp, long int pos)
{
  __libc_lock_lock (dirp->lock);
  (void) __lseek (dirp->fd, pos, SEEK_SET);
  dirp->size = 0;
  dirp->offset = 0;
  dirp->filepos = pos;
  __libc_lock_unlock (dirp->lock);
}
