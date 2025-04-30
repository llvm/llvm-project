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
#include <sys/socket.h>
#include <hurd.h>
#include <hurd/socket.h>
#include <hurd/fd.h>
#include <fcntl.h>
#include <fcntl-internal.h>

/* Create a new socket of type TYPE in domain DOMAIN, using
   protocol PROTOCOL.  If PROTOCOL is zero, one is chosen automatically.
   Returns a file descriptor for the new socket, or -1 for errors.  */
int
__socket (int domain, int type, int protocol)
{
  error_t err;
  socket_t sock, server;
  int flags = sock_to_o_flags (type & ~SOCK_TYPE_MASK);
  type &= SOCK_TYPE_MASK;

  if (flags & ~(O_CLOEXEC | O_NONBLOCK))
    return __hurd_fail (EINVAL);

  /* Find the socket server for DOMAIN.  */
  server = _hurd_socket_server (domain, 0);
  if (server == MACH_PORT_NULL)
    return -1;

  err = __socket_create (server, type, protocol, &sock);
  if (err == MACH_SEND_INVALID_DEST || err == MIG_SERVER_DIED
      || err == MIG_BAD_ID || err == EOPNOTSUPP)
    {
      /* On the first use of the socket server during the operation,
	 allow for the old server port dying.  */
      server = _hurd_socket_server (domain, 1);
      if (server == MACH_PORT_NULL)
	return -1;
      err = __socket_create (server, type, protocol, &sock);
    }

  /* These errors all mean that the server node doesn't support the
     socket.defs protocol, which we'll take to mean that the protocol
     isn't supported.  */
  if (err == MACH_SEND_INVALID_DEST || err == MIG_SERVER_DIED
      || err == MIG_BAD_ID || err == EOPNOTSUPP)
    err = EAFNOSUPPORT;

  if (! err)
    {
      if (flags & O_NONBLOCK)
	err = __io_set_some_openmodes (sock, O_NONBLOCK);
      /* TODO: do we need special ERR massaging after the previous call?  */
    }

  if (err)
    return __hurd_fail (err);

  return _hurd_intern_fd (sock, O_IGNORE_CTTY | flags, 1);
}

libc_hidden_def (__socket)
weak_alias (__socket, socket)
