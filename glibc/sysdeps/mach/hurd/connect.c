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
#include <hurd.h>
#include <hurd/fd.h>
#include <sys/socket.h>
#include <hurd/socket.h>
#include <sys/un.h>
#include <hurd/ifsock.h>
#include <sysdep-cancel.h>
#include "hurd/hurdsocket.h"

/* Open a connection on socket FD to peer at ADDR (which LEN bytes long).
   For connectionless socket types, just set the default address to send to
   and the only address from which to accept transmissions.
   Return 0 on success, -1 for errors.  */
int
__connect (int fd, __CONST_SOCKADDR_ARG addrarg, socklen_t len)
{
  error_t err;
  addr_port_t aport;
  const struct sockaddr_un *addr = addrarg.__sockaddr_un__;
  int cancel_oldtype;

  if (addr->sun_family == AF_LOCAL)
    {
      char *name = _hurd_sun_path_dupa (addr, len);
      /* For the local domain, we must look up the name as a file and talk
	 to it with the ifsock protocol.  */
      file_t file;
      cancel_oldtype = LIBC_CANCEL_ASYNC();
      file = __file_name_lookup (name, 0, 0);
      LIBC_CANCEL_RESET (cancel_oldtype);
      if (file == MACH_PORT_NULL)
	return -1;
      err = __ifsock_getsockaddr (file, &aport);
      __mach_port_deallocate (__mach_task_self (), file);
      if (err == MIG_BAD_ID || err == EOPNOTSUPP)
	/* The file did not grok the ifsock protocol.  */
	err = ENOTSOCK;
      if (err)
	return __hurd_fail (err);
    }
  else
    err = EIEIO;

  err = HURD_DPORT_USE_CANCEL (fd,
			({
			  if (err)
			    err = __socket_create_address (port,
							   addr->sun_family,
							   (char *) addr, len,
							   &aport);
			  if (! err)
			    {
			      cancel_oldtype = LIBC_CANCEL_ASYNC();
			      err = __socket_connect (port, aport);
			      LIBC_CANCEL_RESET (cancel_oldtype);
			      __mach_port_deallocate (__mach_task_self (),
						      aport);
			    }
			  err;
			}));

  return err ? __hurd_dfail (fd, err) : 0;
}

libc_hidden_def (__connect)
weak_alias (__connect, connect)
