/* Send multiple messages on a socket.  Stub version.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

/* Send a VLEN messages as described by VMESSAGES to socket FD.
   Returns the number of datagrams successfully written or -1 for errors.  */
int
__sendmmsg (int fd, struct mmsghdr *vmessages, unsigned int vlen, int flags)
{
  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (__sendmmsg)
weak_alias (__sendmmsg, sendmmsg)
stub_warning (sendmmsg)
