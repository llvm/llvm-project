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
#include <string.h>
#include <sys/socket.h>
#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/socket.h>
#include <sysdep-cancel.h>

/* Read N bytes into BUF through socket FD.
   If ADDR is not NULL, fill in *ADDR_LEN bytes of it with tha address of
   the sender, and store the actual size of the address in *ADDR_LEN.
   Returns the number of bytes read or -1 for errors.  */
ssize_t
__recvfrom (int fd, void *buf, size_t n, int flags, __SOCKADDR_ARG addrarg,
	    socklen_t *addr_len)
{
  error_t err;
  mach_port_t addrport;
  char *bufp = buf;
  mach_msg_type_number_t nread = n;
  mach_port_t *ports;
  mach_msg_type_number_t nports = 0;
  char *cdata = NULL;
  mach_msg_type_number_t clen = 0;
  struct sockaddr *addr = addrarg.__sockaddr__;
  int cancel_oldtype;

  cancel_oldtype = LIBC_CANCEL_ASYNC();
  err = HURD_DPORT_USE_CANCEL (fd, __socket_recv (port, &addrport,
						  flags, &bufp, &nread,
						  &ports, &nports,
						  &cdata, &clen,
						  &flags,
						  n));
  LIBC_CANCEL_RESET (cancel_oldtype);

  if (err)
    return __hurd_sockfail (fd, flags, err);

  /* Get address data for the returned address port if requested.  */
  if (addr != NULL && addrport != MACH_PORT_NULL)
    {
      char *buf = (char *) addr;
      mach_msg_type_number_t buflen = *addr_len;
      int type;

      cancel_oldtype = LIBC_CANCEL_ASYNC();
      err = __socket_whatis_address (addrport, &type, &buf, &buflen);
      LIBC_CANCEL_RESET (cancel_oldtype);
      if (err == EOPNOTSUPP)
	/* If the protocol server can't tell us the address, just return a
	   zero-length one.  */
	{
	  buf = (char *)addr;
	  buflen = 0;
	  err = 0;
	}

      if (err)
	{
	  __mach_port_deallocate (__mach_task_self (), addrport);
	  return __hurd_sockfail (fd, flags, err);
	}

      if (*addr_len > buflen)
	*addr_len = buflen;

      if (buf != (char *) addr)
	{
	  memcpy (addr, buf, *addr_len);
	  __vm_deallocate (__mach_task_self (), (vm_address_t) buf, buflen);
	}

      if (buflen > 0)
	addr->sa_family = type;
    }
  else if (addr_len != NULL)
    *addr_len = 0;

  __mach_port_deallocate (__mach_task_self (), addrport);

  /* Toss control data; we don't care.  */
  __vm_deallocate (__mach_task_self (), (vm_address_t) cdata, clen);

  if (bufp != buf)
    {
      memcpy (buf, bufp, nread);
      __vm_deallocate (__mach_task_self (), (vm_address_t) bufp, nread);
    }

  return nread;
}

weak_alias (__recvfrom, recvfrom)
