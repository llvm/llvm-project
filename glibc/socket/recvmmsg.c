/* Receive multiple messages on a socket.  Stub version.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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
#include <sys/socket.h>

/* Receive up to VLEN messages as described by VMESSAGES from socket FD.
   Returns the number of bytes read or -1 for errors.  */
int
recvmmsg (int fd, struct mmsghdr *vmessages, unsigned int vlen, int flags,
	  struct timespec *tmo)
{
  __set_errno (ENOSYS);
  return -1;
}
stub_warning (recvmmsg)
