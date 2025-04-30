/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <sys/socket.h>
#include <sysdep-cancel.h>
#include <socketcall.h>

ssize_t
__libc_recv (int fd, void *buf, size_t len, int flags)
{
#ifdef __ASSUME_RECV_SYSCALL
  return SYSCALL_CANCEL (recv, fd, buf, len, flags);
#elif defined __ASSUME_RECVFROM_SYSCALL
  return SYSCALL_CANCEL (recvfrom, fd, buf, len, flags, NULL, NULL);
#else
  return SOCKETCALL_CANCEL (recv, fd, buf, len, flags);
#endif
}
weak_alias (__libc_recv, recv)
weak_alias (__libc_recv, __recv)
libc_hidden_weak (__recv)
