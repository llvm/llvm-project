/* Catastrophic failure reports.  Linux version.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <sys/uio.h>
#include <stdbool.h>
#include <sysdep.h>

static bool
writev_for_fatal (int fd, const struct iovec *iov, size_t niov, size_t total)
{
  ssize_t cnt;
  do
    cnt = INTERNAL_SYSCALL_CALL (writev, fd, iov, niov);
  while (INTERNAL_SYSCALL_ERROR_P (cnt)
         && INTERNAL_SYSCALL_ERRNO (cnt) == EINTR);
  return cnt == total;
}
#define WRITEV_FOR_FATAL	writev_for_fatal

#include <sysdeps/posix/libc_fatal.c>
