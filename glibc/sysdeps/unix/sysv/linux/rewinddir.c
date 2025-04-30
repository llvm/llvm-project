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

#include <stddef.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirstream.h>

/* Rewind DIRP to the beginning of the directory.  */
void
__rewinddir (DIR *dirp)
{
#if IS_IN (libc)
  __libc_lock_lock (dirp->lock);
#endif
  (void) __lseek (dirp->fd, (off_t) 0, SEEK_SET);
  dirp->filepos = 0;
  dirp->offset = 0;
  dirp->size = 0;
  dirp->errcode = 0;
#if IS_IN (libc)
  __libc_lock_unlock (dirp->lock);
#endif
}
libc_hidden_def (__rewinddir)
weak_alias (__rewinddir, rewinddir)
