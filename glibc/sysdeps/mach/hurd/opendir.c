/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <not-cancel.h>
#include "dirstream.h"


/* Open a directory stream on a file descriptor in Hurd internal form.
   We do no checking here on the descriptor.  */
DIR *
_hurd_fd_opendir (struct hurd_fd *d)
{
  DIR *dirp;

  if (d == NULL)
    {
      errno = EBADF;
      return NULL;
    }

  dirp = (DIR *) malloc (sizeof (DIR));
  if (dirp == NULL)
    return NULL;

  /* Set the descriptor to close on exec. */
  HURD_CRITICAL_BEGIN;
  __spin_lock (&d->port.lock);
  d->flags |= FD_CLOEXEC;
  __spin_unlock (&d->port.lock);
  HURD_CRITICAL_END;

  dirp->__fd = d;
  dirp->__data = dirp->__ptr = NULL;
  dirp->__entry_data = dirp->__entry_ptr = 0;
  dirp->__allocation = 0;
  dirp->__size = 0;

  __libc_lock_init (dirp->__lock);

  return dirp;
}


DIR *
__opendirat (int dfd, const char *name)
{
  if (name[0] == '\0')
    {
      /* POSIX.1-1990 says an empty name gets ENOENT;
	 but `open' might like it fine.  */
      __set_errno (ENOENT);
      return NULL;
    }

  int flags = O_RDONLY | O_NONBLOCK | O_DIRECTORY | O_CLOEXEC;
  int fd;
#if IS_IN (rtld)
  assert (dfd == AT_FDCWD);
  fd = __open_nocancel (name, flags);
#else
  fd = __openat_nocancel (dfd, name, flags);
#endif
  if (fd < 0)
    return NULL;

  /* Extract the pointer to the descriptor structure.  */
  DIR *dirp = _hurd_fd_opendir (_hurd_fd_get (fd));
  if (dirp == NULL)
    __close (fd);

  return dirp;
}


/* Open a directory stream on NAME.  */
DIR *
__opendir (const char *name)
{
#if 0 /* TODO.  */
  return __opendirat (AT_FDCWD, name);
#else
  if (name[0] == '\0')
    {
      /* POSIX.1-1990 says an empty name gets ENOENT;
	 but `open' might like it fine.  */
      __set_errno (ENOENT);
      return NULL;
    }

  int fd = __open (name, O_RDONLY | O_NONBLOCK | O_DIRECTORY);
  if (fd < 0)
    return NULL;

  /* Extract the pointer to the descriptor structure.  */
  DIR *dirp = _hurd_fd_opendir (_hurd_fd_get (fd));
  if (dirp == NULL)
    __close (fd);

  return dirp;
#endif
}
weak_alias (__opendir, opendir)
