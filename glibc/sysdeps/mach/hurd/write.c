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

#include <sysdep-cancel.h>
#include <not-cancel.h>

ssize_t
__libc_write (int fd, const void *buf, size_t nbytes)
{
  ssize_t ret;
  int cancel_oldtype = LIBC_CANCEL_ASYNC();
  ret = __write_nocancel (fd, buf, nbytes);
  LIBC_CANCEL_RESET (cancel_oldtype);
  return ret;
}
libc_hidden_def (__libc_write)
weak_alias (__libc_write, __write)
libc_hidden_weak (__write)
weak_alias (__libc_write, write)
libc_hidden_weak (write)
