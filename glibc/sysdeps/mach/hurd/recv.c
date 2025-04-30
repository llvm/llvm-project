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
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/socket.h>
#include <string.h>
#include <sysdep-cancel.h>

/* Read N bytes into BUF from socket FD.
   Returns the number read or -1 for errors.  */

ssize_t
__recv (int fd, void *buf, size_t n, int flags)
{
  error_t err;
  mach_port_t addrport;
  char *bufp = buf;
  mach_msg_type_number_t nread = n;
  mach_port_t *ports;
  mach_msg_type_number_t nports = 0;
  char *cdata = NULL;
  mach_msg_type_number_t clen = 0;
  int cancel_oldtype;

  cancel_oldtype = LIBC_CANCEL_ASYNC();
  err = HURD_DPORT_USE_CANCEL (fd, __socket_recv (port, &addrport,
						  flags, &bufp, &nread,
						  &ports, &nports,
						  &cdata, &clen,
						  &flags,
						  n));
  LIBC_CANCEL_RESET (cancel_oldtype);

  if (err == MIG_BAD_ID || err == EOPNOTSUPP)
    /* The file did not grok the socket protocol.  */
    err = ENOTSOCK;
  if (err)
    return __hurd_sockfail (fd, flags, err);

  __mach_port_deallocate (__mach_task_self (), addrport);
  __vm_deallocate (__mach_task_self (), (vm_address_t) cdata, clen);

  if (bufp != buf)
    {
      memcpy (buf, bufp, nread);
      __vm_deallocate (__mach_task_self (), (vm_address_t) bufp, nread);
    }

  return nread;
}
libc_hidden_def (__recv)
weak_alias (__recv, recv)
