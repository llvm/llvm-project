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

#include <errno.h>
#include <fcntl.h>
#include <sysdep-cancel.h>

#ifndef __OFF_T_MATCHES_OFF64_T
/* Reserve storage for the data of the file associated with FD.  */
int
fallocate (int fd, int mode, __off_t offset, __off_t len)
{
  return SYSCALL_CANCEL (fallocate, fd, mode,
			 SYSCALL_LL (offset), SYSCALL_LL (len));
}
#endif
