/* Linux recvmsg syscall wrapper.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
__libc_recvmsg (int fd, struct msghdr *msg, int flags)
{
  ssize_t r;
#ifndef __ASSUME_TIME64_SYSCALLS
  socklen_t orig_controllen = msg != NULL ? msg->msg_controllen : 0;
#endif

#ifdef __ASSUME_RECVMSG_SYSCALL
  r = SYSCALL_CANCEL (recvmsg, fd, msg, flags);
#else
  r = SOCKETCALL_CANCEL (recvmsg, fd, msg, flags);
#endif

#ifndef __ASSUME_TIME64_SYSCALLS
  if (r >= 0 && orig_controllen != 0)
    __convert_scm_timestamps (msg, orig_controllen);
#endif

  return r;
}
weak_alias (__libc_recvmsg, recvmsg)
weak_alias (__libc_recvmsg, __recvmsg)
#if __TIMESIZE != 64
weak_alias (__recvmsg, __recvmsg64)
#endif
