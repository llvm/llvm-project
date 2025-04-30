/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include <sys/un.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/ifsock.h>
#include <hurd/socket.h>
#include <sysdep-cancel.h>
#include "hurd/hurdsocket.h"

/* Send N bytes of BUF on socket FD to peer at address ADDR (which is
   ADDR_LEN bytes long).  Returns the number sent, or -1 for errors.  */
ssize_t
__sendto (int fd,
	  const void *buf,
	  size_t n,
	  int flags,
	  const struct sockaddr_un *addr,
	  socklen_t addr_len)
{
  addr_port_t aport = MACH_PORT_NULL;
  error_t err;
  size_t wrote;

  /* Get an address port for the desired destination address.  */
  error_t create_address_port (io_t port,
			       const struct sockaddr_un *addr,
			       socklen_t addr_len,
			       addr_port_t *aport)
    {
      error_t err_port;

      if (addr->sun_family == AF_LOCAL)
	{
	  char *name = _hurd_sun_path_dupa (addr, addr_len);
	  /* For the local domain, we must look up the name as a file and talk
	     to it with the ifsock protocol.  */
	  file_t file = __file_name_lookup (name, 0, 0);
	  if (file == MACH_PORT_NULL)
	    return errno;
	  err_port = __ifsock_getsockaddr (file, aport);
	  __mach_port_deallocate (__mach_task_self (), file);
	  if (err_port == MIG_BAD_ID || err_port == EOPNOTSUPP)
	    /* The file did not grok the ifsock protocol.  */
	    err_port = ENOTSOCK;
	}
      else
	{
	  err_port = __socket_create_address (port,
					      addr->sun_family,
					      (char *) addr,
					      addr_len,
					      aport);
	}

      return err_port;
    }

  err = HURD_DPORT_USE_CANCEL (fd,
			({
			  if (addr != NULL)
			    err = create_address_port (port, addr, addr_len,
						       &aport);
			  else
			    err = 0;
			  if (! err)
			    {
			      /* Send the data.  */
			      int cancel_oldtype = LIBC_CANCEL_ASYNC();
			      err = __socket_send (port, aport,
						   flags, buf, n,
						   NULL,
						   MACH_MSG_TYPE_COPY_SEND, 0,
						   NULL, 0, &wrote);
			      LIBC_CANCEL_RESET (cancel_oldtype);
			    }
			  err;
			}));

  if (aport != MACH_PORT_NULL)
    __mach_port_deallocate (__mach_task_self (), aport);

  return err ? __hurd_sockfail (fd, flags, err) : wrote;
}

weak_alias (__sendto, sendto)
