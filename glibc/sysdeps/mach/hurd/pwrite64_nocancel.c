/* Write block to given position in file without changing file pointer.
   Hurd version.
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

#include <errno.h>
#include <unistd.h>
#include <hurd/fd.h>
#include <not-cancel.h>

ssize_t
__pwrite64_nocancel (int fd, const void *buf, size_t nbytes, off64_t offset)
{
  error_t err;
  if (offset < 0)
    err = EINVAL;
  else
    err = HURD_FD_USE (fd, _hurd_fd_write (descriptor, buf, &nbytes, offset));
  return err ? __hurd_dfail (fd, err) : nbytes;
}
libc_hidden_weak (__pwrite64_nocancel)
