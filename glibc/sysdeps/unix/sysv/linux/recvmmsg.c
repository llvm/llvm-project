/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@redhat.com>, 2010.

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
#include <sysdep.h>
#include <socketcall.h>

int
__recvmmsg64 (int fd, struct mmsghdr *vmessages, unsigned int vlen, int flags,
	      struct __timespec64 *timeout)
{
#ifndef __NR_recvmmsg_time64
# define __NR_recvmmsg_time64 __NR_recvmmsg
#endif
  int r = SYSCALL_CANCEL (recvmmsg_time64, fd, vmessages, vlen, flags,
			  timeout);
#ifndef __ASSUME_TIME64_SYSCALLS
  if (r >= 0 || errno != ENOSYS)
    return r;

  struct timespec ts32, *pts32 = NULL;
  if (timeout != NULL)
    {
      if (! in_time_t_range (timeout->tv_sec))
	{
	  __set_errno (EINVAL);
	  return -1;
	}
      ts32 = valid_timespec64_to_timespec (*timeout);
      pts32 = &ts32;
    }

  socklen_t csize[IOV_MAX];
  if (vlen > IOV_MAX)
    vlen = IOV_MAX;
  for (int i = 0; i < vlen; i++)
    csize[i] = vmessages[i].msg_hdr.msg_controllen;

# ifdef __ASSUME_RECVMMSG_SYSCALL
  r = SYSCALL_CANCEL (recvmmsg, fd, vmessages, vlen, flags, pts32);
# else
  r = SOCKETCALL_CANCEL (recvmmsg, fd, vmessages, vlen, flags, pts32);
# endif
  if (r >= 0)
    {
      if (timeout != NULL)
        *timeout = valid_timespec_to_timespec64 (ts32);

      for (int i=0; i < r; i++)
        __convert_scm_timestamps (&vmessages[i].msg_hdr, csize[i]);
    }
#endif /* __ASSUME_TIME64_SYSCALLS  */
  return r;
}
#if __TIMESIZE != 64
libc_hidden_def (__recvmmsg64)

int
__recvmmsg (int fd, struct mmsghdr *vmessages, unsigned int vlen, int flags,
	    struct timespec *timeout)
{
  struct __timespec64 ts64, *pts64 = NULL;
  if (timeout != NULL)
    {
      ts64 = valid_timespec_to_timespec64 (*timeout);
      pts64 = &ts64;
    }
  int r = __recvmmsg64 (fd, vmessages, vlen, flags, pts64);
  if (r >= 0 && timeout != NULL)
    /* The remanining timeout will be always less the input TIMEOUT.  */
    *timeout = valid_timespec64_to_timespec (ts64);
  return r;
}
#endif
weak_alias (__recvmmsg, recvmmsg)
