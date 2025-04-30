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

int
__libc_accept (int fd, __SOCKADDR_ARG addr, socklen_t *len)
{
#ifdef __ASSUME_ACCEPT_SYSCALL
  return SYSCALL_CANCEL (accept, fd, addr.__sockaddr__, len);
#elif defined __ASSUME_ACCEPT4_SYSCALL
  return SYSCALL_CANCEL (accept4, fd, addr.__sockaddr__, len, 0);
#else
  return SOCKETCALL_CANCEL (accept, fd, addr.__sockaddr__, len);
#endif
}
weak_alias (__libc_accept, accept)
libc_hidden_def (accept)
