/* MIPS n32 lseek implementation.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <unistd.h>
#include <sys/types.h>
#include <errno.h>

off_t
__lseek (int fd, off_t offset, int whence)
{
  off64_t res = __lseek64 (fd, offset, whence);
  if (res != (off_t) res)
    {
      __set_errno (EOVERFLOW);
      return (off_t) -1;
    }
  return (off_t) res;
}
libc_hidden_def (__lseek)
weak_alias (__lseek, lseek)
strong_alias (__lseek, __libc_lseek)
