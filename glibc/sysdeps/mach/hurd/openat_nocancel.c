/* openat -- Open a file named relative to an open directory.  Hurd version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/stat.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <not-cancel.h>

/* Open FILE with access OFLAG.  Interpret relative paths relative to
   the directory associated with FD.  If O_CREAT or O_TMPFILE is in OFLAG, a
   third argument is the file protection.  */
int
__openat_nocancel (int fd, const char *file, int oflag, ...)
{
  mode_t mode;
  io_t port;

  if (__OPEN_NEEDS_MODE (oflag))
    {
      va_list arg;
      va_start (arg, oflag);
      mode = va_arg (arg, mode_t);
      va_end (arg);
    }
  else
    mode = 0;

  port = __file_name_lookup_at (fd, 0, file, oflag, mode);
  if (port == MACH_PORT_NULL)
    return -1;

  return _hurd_intern_fd (port, oflag, 1);
}

libc_hidden_def (__openat_nocancel)
