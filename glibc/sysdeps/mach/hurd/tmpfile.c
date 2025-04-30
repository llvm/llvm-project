/* Open a stdio stream on an anonymous temporary file.  Hurd version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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
#include <hurd/fs.h>
#include <hurd/fd.h>
#include <fcntl.h>
#include <unistd.h>
#include <iolibio.h>

/* This returns a new stream opened on a temporary file (generated
   by tmpnam).  The file is opened with mode "w+b" (binary read/write).
   If we couldn't generate a unique filename or the file couldn't
   be opened, NULL is returned.  */
FILE *
__tmpfile (void)
{
  error_t err;
  file_t file;
  int fd;
  FILE *f;

  /* Get a port to the directory that will contain the file.  */
  const char *dirname = __libc_secure_getenv ("TMPDIR") ?: P_tmpdir;
  file_t dir = __file_name_lookup (dirname, 0, 0);
  if (dir == MACH_PORT_NULL)
    return NULL;

  /* Create an unnamed file in the temporary directory.  */
  err = __dir_mkfile (dir, O_RDWR, S_IRUSR | S_IWUSR, &file);
  __mach_port_deallocate (__mach_task_self (), dir);
  if (err)
    return __hurd_fail (err), NULL;

  /* Get a file descriptor for that port.  POSIX.1 requires that streams
     returned by tmpfile allocate file descriptors as fopen would.  */
  fd = _hurd_intern_fd (file, O_RDWR, 1); /* dealloc on error */
  if (fd < 0)
    return NULL;

  /* Open a stream on the unnamed file.
     It will cease to exist when this stream is closed.  */
  if ((f = _IO_fdopen (fd, "w+b")) == NULL)
    __close (fd);

  return f;
}

#include <shlib-compat.h>
versioned_symbol (libc, __tmpfile, tmpfile, GLIBC_2_1);

weak_alias (__tmpfile, tmpfile64)
