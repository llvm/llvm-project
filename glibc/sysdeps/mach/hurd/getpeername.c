/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <sys/socket.h>

#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/socket.h>

/* Put the address of the peer connected to socket FD into *ADDR
   (which is *LEN bytes long), and its actual length into *LEN.  */
int
__getpeername (int fd, __SOCKADDR_ARG addrarg, socklen_t *len)
{
  error_t err;
  mach_msg_type_number_t buflen = *len;
  int type;
  struct sockaddr *addr = addrarg.__sockaddr__;
  char *buf = (char *) addr;
  addr_port_t aport;

  if (err = HURD_DPORT_USE (fd, __socket_peername (port, &aport)))
    return __hurd_dfail (fd, err);

  err = __socket_whatis_address (aport, &type, &buf, &buflen);
  __mach_port_deallocate (__mach_task_self (), aport);

  if (err)
    return __hurd_dfail (fd, err);

  if (*len > buflen)
    *len = buflen;

  if (buf != (char *) addr)
    {
      memcpy (addr, buf, *len);
      __vm_deallocate (__mach_task_self (), (vm_address_t) buf, buflen);
    }

  const sa_family_t family = type;
  if (*len > offsetof (struct sockaddr, sa_family))
    {
      if (*len < (char *) (&addr->sa_family + 1) - (char *) addr)
	memcpy (&addr->sa_family, &family,
		*len - offsetof (struct sockaddr, sa_family));
      else
	addr->sa_family = family;
    }

  return 0;
}

weak_alias (__getpeername, getpeername)
