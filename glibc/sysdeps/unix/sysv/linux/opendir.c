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

#include <dirent.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>	/* For BUFSIZ.  */
#include <sys/param.h>	/* For MIN and MAX.  */

#include <not-cancel.h>

enum {
  opendir_oflags = O_RDONLY|O_NDELAY|O_DIRECTORY|O_LARGEFILE|O_CLOEXEC
};

static bool
invalid_name (const char *name)
{
  if (__glibc_unlikely (name[0] == '\0'))
    {
      /* POSIX.1-1990 says an empty name gets ENOENT;
	 but `open' might like it fine.  */
      __set_errno (ENOENT);
      return true;
    }
  return false;
}

static DIR *
opendir_tail (int fd)
{
  if (__glibc_unlikely (fd < 0))
    return NULL;

  /* Now make sure this really is a directory and nothing changed since the
     `stat' call.  The S_ISDIR check is superfluous if O_DIRECTORY works,
     but it's cheap and we need the stat call for st_blksize anyway.  */
  struct __stat64_t64 statbuf;
  if (__glibc_unlikely (__fstat64_time64 (fd, &statbuf) < 0))
    goto lose;
  if (__glibc_unlikely (! S_ISDIR (statbuf.st_mode)))
    {
      __set_errno (ENOTDIR);
    lose:
      __close_nocancel_nostatus (fd);
      return NULL;
    }

  return __alloc_dir (fd, true, 0, &statbuf);
}


#if IS_IN (libc)
DIR *
__opendirat (int dfd, const char *name)
{
  if (__glibc_unlikely (invalid_name (name)))
    return NULL;

  return opendir_tail (__openat_nocancel (dfd, name, opendir_oflags));
}
#endif


/* Open a directory stream on NAME.  */
DIR *
__opendir (const char *name)
{
  if (__glibc_unlikely (invalid_name (name)))
    return NULL;

  return opendir_tail (__open_nocancel (name, opendir_oflags));
}
weak_alias (__opendir, opendir)

DIR *
__alloc_dir (int fd, bool close_fd, int flags,
	     const struct __stat64_t64 *statp)
{
  /* We have to set the close-on-exit flag if the user provided the
     file descriptor.  */
  if (!close_fd
      && __glibc_unlikely (__fcntl64_nocancel (fd, F_SETFD, FD_CLOEXEC) < 0))
    return NULL;

  /* The st_blksize value of the directory is used as a hint for the
     size of the buffer which receives struct dirent values from the
     kernel.  st_blksize is limited to max_buffer_size, in case the
     file system provides a bogus value.  */
  enum { max_buffer_size = 1048576 };

  enum { allocation_size = 32768 };
  _Static_assert (allocation_size >= sizeof (struct dirent64),
		  "allocation_size < sizeof (struct dirent64)");

  /* Increase allocation if requested, but not if the value appears to
     be bogus.  It will be between 32Kb and 1Mb.  */
  size_t allocation = MIN (MAX ((size_t) statp->st_blksize, allocation_size),
			   max_buffer_size);

  DIR *dirp = (DIR *) malloc (sizeof (DIR) + allocation);
  if (dirp == NULL)
    {
      if (close_fd)
	__close_nocancel_nostatus (fd);
      return NULL;
    }

  dirp->fd = fd;
#if IS_IN (libc)
  __libc_lock_init (dirp->lock);
#endif
  dirp->allocation = allocation;
  dirp->size = 0;
  dirp->offset = 0;
  dirp->filepos = 0;
  dirp->errcode = 0;

  return dirp;
}
