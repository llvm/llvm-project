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

#include <unistd.h>
#include <sys/types.h>
#include <errno.h>

/* Seek to OFFSET on FD, starting from WHENCE.  */
off_t
__libc_lseek (int fd, off_t offset, int whence)
{
  off64_t res64 = __libc_lseek64 (fd, (off64_t) offset, whence);
  off_t res = (off_t) res64;

  if (sizeof res != sizeof res64 && res != res64)
    {
      __set_errno (EOVERFLOW);
      return (off_t) -1;
    }

  return res;
}

weak_alias (__libc_lseek, __lseek)
libc_hidden_def (__lseek)
weak_alias (__libc_lseek, lseek)
