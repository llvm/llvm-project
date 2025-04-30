/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include <fcntl.h>
#include <sysdep.h>

extern int __posix_fallocate64_l64 (int fd, __off64_t offset, __off64_t len);
libc_hidden_proto (__posix_fallocate64_l64)
#define __posix_fallocate64_l64 static internal_fallocate64
#include <sysdeps/posix/posix_fallocate64.c>
#undef __posix_fallocate64_l64

/* Reserve storage for the data of the file associated with FD.  */
int
__posix_fallocate64_l64 (int fd, __off64_t offset, __off64_t len)
{
  int res = INTERNAL_SYSCALL_CALL (fallocate, fd, 0,
				   SYSCALL_LL64 (offset), SYSCALL_LL64 (len));
  if (! INTERNAL_SYSCALL_ERROR_P (res))
    return 0;
  if (INTERNAL_SYSCALL_ERRNO (res) != EOPNOTSUPP)
    return INTERNAL_SYSCALL_ERRNO (res);
  return internal_fallocate64 (fd, offset, len);
}
libc_hidden_def (__posix_fallocate64_l64)
