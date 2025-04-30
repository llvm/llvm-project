/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Chris Metcalf <cmetcalf@tilera.com>, 2011.

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

#include <fcntl.h>
#include <unistd.h>
#include <sysdep.h>

/* Duplicate FD to FD2, closing the old FD2 and making FD2 be
   open the same file as FD is.  Return FD2 or -1.  */
int
__dup2 (int fd, int fd2)
{
  /* For the degenerate case, check if the fd is valid (by trying to
     get the file status flags) and return it, or else return EBADF.  */
  if (fd == fd2)
    return __libc_fcntl (fd, F_GETFL, 0) < 0 ? -1 : fd;

  return INLINE_SYSCALL (dup3, 3, fd, fd2, 0);
}
libc_hidden_def (__dup2)
weak_alias (__dup2, dup2)
