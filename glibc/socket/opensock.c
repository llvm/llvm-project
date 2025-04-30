/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <sys/socket.h>
#include <libc-lock.h>

/* Return a socket of any type.  The socket can be used in subsequent
   ioctl calls to talk to the kernel.  */
int
__opensock (void)
{
  /* Cache the last AF that worked, to avoid many redundant calls to
     socket().  */
  static int sock_af = -1;
  int fd = -1;
  __libc_lock_define_initialized (static, lock);

  if (sock_af != -1)
    {
      fd = __socket (sock_af, SOCK_DGRAM, 0);
      if (fd != -1)
        return fd;
    }

  __libc_lock_lock (lock);

  if (sock_af != -1)
    fd = __socket (sock_af, SOCK_DGRAM, 0);

  if (fd == -1)
    {
#ifdef AF_INET
      fd = __socket (sock_af = AF_INET, SOCK_DGRAM, 0);
#endif
#ifdef AF_INET6
      if (fd < 0)
	fd = __socket (sock_af = AF_INET6, SOCK_DGRAM, 0);
#endif
#ifdef AF_IPX
      if (fd < 0)
	fd = __socket (sock_af = AF_IPX, SOCK_DGRAM, 0);
#endif
#ifdef AF_AX25
      if (fd < 0)
	fd = __socket (sock_af = AF_AX25, SOCK_DGRAM, 0);
#endif
#ifdef AF_APPLETALK
      if (fd < 0)
	fd = __socket (sock_af = AF_APPLETALK, SOCK_DGRAM, 0);
#endif
    }

  __libc_lock_unlock (lock);
  return fd;
}
