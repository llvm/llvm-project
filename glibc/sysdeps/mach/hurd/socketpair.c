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
#include <fcntl.h>
#include <fcntl-internal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/socket.h>

/* Create two new sockets, of type TYPE in domain DOMAIN and using
   protocol PROTOCOL, which are connected to each other, and put file
   descriptors for them in FDS[0] and FDS[1].  If PROTOCOL is zero,
   one will be chosen automatically.  Returns 0 on success, -1 for errors.  */
int
__socketpair (int domain, int type, int protocol, int fds[2])
{
  error_t err;
  socket_t server, sock1, sock2;
  int d1, d2;
  int flags = sock_to_o_flags (type & ~SOCK_TYPE_MASK);
  type &= SOCK_TYPE_MASK;

  if (flags & ~(O_CLOEXEC | O_NONBLOCK))
    return __hurd_fail (EINVAL);

  if (fds == NULL)
    return __hurd_fail (EINVAL);

  /* Find the domain's socket server.  */
  server = _hurd_socket_server (domain, 0);
  if (server == MACH_PORT_NULL)
    return -1;

  /* Create two sockets and connect them together.  */

  err = __socket_create (server, type, protocol, &sock1);
  if (err == MACH_SEND_INVALID_DEST || err == MIG_SERVER_DIED
      || err == MIG_BAD_ID || err == EOPNOTSUPP)
    {
      /* On the first use of the socket server during the operation,
	 allow for the old server port dying.  */
      server = _hurd_socket_server (domain, 1);
      if (server == MACH_PORT_NULL)
	return -1;
      err = __socket_create (server, type, protocol, &sock1);
    }
  /* TODO: do we need special ERR massaging here, like it is done in
     __socket?  */
  if (! err)
    {
      if (flags & O_NONBLOCK)
	err = __io_set_some_openmodes (sock1, O_NONBLOCK);
      /* TODO: do we need special ERR massaging after the previous call?  */
    }
  if (err)
    return __hurd_fail (err);
  if (err = __socket_create (server, type, protocol, &sock2))
    {
      __mach_port_deallocate (__mach_task_self (), sock1);
      return __hurd_fail (err);
    }
  if (flags & O_NONBLOCK)
    err = __io_set_some_openmodes (sock2, O_NONBLOCK);
  /* TODO: do we need special ERR massaging after the previous call?  */
  if (! err)
    err = __socket_connect2 (sock1, sock2);
  if (err)
    {
      __mach_port_deallocate (__mach_task_self (), sock1);
      __mach_port_deallocate (__mach_task_self (), sock2);
      return __hurd_fail (err);
    }

  /* Put the sockets into file descriptors.  */

  d1 = _hurd_intern_fd (sock1, O_IGNORE_CTTY | flags, 1);
  if (d1 < 0)
    {
      __mach_port_deallocate (__mach_task_self (), sock2);
      return -1;
    }
  d2 = _hurd_intern_fd (sock2, O_IGNORE_CTTY | flags, 1);
  if (d2 < 0)
    {
      err = errno;
      (void) __close (d1);
      return __hurd_fail (err);
    }

  fds[0] = d1;
  fds[1] = d2;
  return 0;
}

weak_alias (__socketpair, socketpair)
