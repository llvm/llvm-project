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
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <dirstream.h>
#include <not-cancel.h>


/* Close the directory stream DIRP.
   Return 0 if successful, -1 if not.  */
int
__closedir (DIR *dirp)
{
  int fd;

  if (dirp == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  /* We do not try to synchronize access here.  If some other thread
     still uses this handle it is a big mistake and that thread
     deserves all the bad data it gets.  */

  fd = dirp->fd;

#if IS_IN (libc)
  __libc_lock_fini (dirp->lock);
#endif

  free ((void *) dirp);

  return __close_nocancel (fd);
}
weak_alias (__closedir, closedir)
