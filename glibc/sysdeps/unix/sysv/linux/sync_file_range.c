/* Selective file content synch'ing.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#include <sysdep-cancel.h>

int
sync_file_range (int fd, __off64_t offset, __off64_t len, unsigned int flags)
{
#if defined (__NR_sync_file_range2)
  return SYSCALL_CANCEL (sync_file_range2, fd, flags, SYSCALL_LL64 (offset),
			 SYSCALL_LL64 (len));
#elif defined (__NR_sync_file_range)
  return SYSCALL_CANCEL (sync_file_range, fd,
			 __ALIGNMENT_ARG SYSCALL_LL64 (offset),
			 SYSCALL_LL64 (len), flags);
#endif
}
