/* Write block at given position in file without changing file pointer.
   Hurd version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

/* Write NBYTES of BUF to FD at given position OFFSET without changing
   the file position.  Return the number written, or -1.  */
ssize_t
__libc_pwrite (int fd, const void *buf, size_t nbytes, off_t offset)
{
  return __libc_pwrite64 (fd, buf, nbytes, (off64_t) offset);
}

#ifndef __libc_pwrite
strong_alias (__libc_pwrite, __pwrite)
weak_alias (__libc_pwrite, pwrite)
#endif
