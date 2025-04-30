/* Open a directory stream from a file descriptor.  Hurd version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
#include <errno.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <fcntl.h>

DIR *_hurd_fd_opendir (struct hurd_fd *d); /* opendir.c */

/* Open a directory stream on FD.  */
DIR *
__fdopendir (int fd)
{
  struct hurd_fd *d = _hurd_fd_get (fd);

  if (d == NULL)
    {
      errno = EBADF;
      return NULL;
    }

  /* Ensure that it's a directory.  */
  error_t err = HURD_FD_PORT_USE
    (d, ({
	file_t dir = __file_name_lookup_under (port, "./",
					       O_DIRECTORY | O_NOTRANS, 0);;
	if (dir != MACH_PORT_NULL)
	  __mach_port_deallocate (__mach_task_self (), dir);
	dir != MACH_PORT_NULL ? 0 : errno;
      }));

  if (err)
    {
      errno = err;
      return NULL;
    }

  return _hurd_fd_opendir (d);
}
weak_alias (__fdopendir, fdopendir)
