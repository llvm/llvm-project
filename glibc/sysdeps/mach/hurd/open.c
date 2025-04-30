/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <sysdep-cancel.h>

/* Open FILE with access OFLAG.  If O_CREAT or O_TMPFILE is in OFLAG,
   a third argument is the file protection.  */
int
__libc_open (const char *file, int oflag, ...)
{
  mode_t mode;
  io_t port;
  int cancel_oldtype;

  if (__OPEN_NEEDS_MODE (oflag))
    {
      va_list arg;
      va_start (arg, oflag);
      mode = va_arg (arg, mode_t);
      va_end (arg);
    }
  else
    mode = 0;

  cancel_oldtype = LIBC_CANCEL_ASYNC();
  port = __file_name_lookup (file, oflag, mode);
  LIBC_CANCEL_RESET (cancel_oldtype);

  if (port == MACH_PORT_NULL)
    return -1;

  return _hurd_intern_fd (port, oflag, 1);
}

libc_hidden_def (__libc_open)
weak_alias (__libc_open, __open)
libc_hidden_weak (__open)
weak_alias (__libc_open, open)


/* open64 is just the same as open for us.  */
weak_alias (__libc_open, __libc_open64)
weak_alias (__libc_open, __open64)
libc_hidden_weak (__open64)
weak_alias (__libc_open, open64)
