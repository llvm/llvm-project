/* Implementation of the _dl_write function.  Linux version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <unistd.h>
#include <ldsodefs.h>

ssize_t
_dl_write (int fd, const void *buffer, size_t length)
{
  long int r = INTERNAL_SYSCALL_CALL (write, fd, buffer, length);
  if (INTERNAL_SYSCALL_ERROR_P (r))
    r = - INTERNAL_SYSCALL_ERRNO (r);
  return r;
}
