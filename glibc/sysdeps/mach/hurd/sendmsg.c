/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <hurd.h>
#include <hurd/fd.h>
#include <hurd/ifsock.h>
#include <hurd/socket.h>
#include <sysdep-cancel.h>
#include "hurd/hurdsocket.h"

/* Send a message described MESSAGE on socket FD.
   Returns the number of bytes sent, or -1 for errors.  */
ssize_t
__libc_sendmsg (int fd, const struct msghdr *message, int flags)
{
  error_t err = 0;
  struct cmsghdr *cmsg;
  mach_port_t *ports = NULL;
  mach_msg_type_number_t nports = 0;
  int *fds, nfds;
  struct sockaddr_un *addr = message->msg_name;
  socklen_t addr_len = message->msg_namelen;
  addr_port_t aport = MACH_PORT_NULL;
  union
  {
    char *ptr;
    vm_address_t addr;
  } data = { .ptr = NULL };
  char data_buf[2048];
  mach_msg_type_number_t len;
  mach_msg_type_number_t amount;
  int dealloc = 0;
  int socketrpc = 0;
  int i;

  /* Find the total number of bytes to be written.  */
  len = 0;
  for (i = 0; i < message->msg_iovlen; i++)
    {
      if (message->msg_iov[i].iov_len > 0)
	{
	  /* As an optimization, if we only have a single non-empty
             iovec, we set DATA and LEN from it.  */
	  if (len == 0)
	    data.ptr = message->msg_iov[i].iov_base;
	  else
	    data.ptr = NULL;

	  len += message->msg_iov[i].iov_len;
	}
    }

  if (data.ptr == NULL)
    {
      size_t to_copy;
      char *buf;

      /* Allocate a temporary buffer to hold the data.  For small
         amounts of data, we allocate a buffer on the stack.  Larger
         amounts of data are stored in a page-aligned buffer.  The
         limit of 2048 bytes is inspired by the MiG stubs.  */
      if (len > 2048)
	{
	  err = __vm_allocate (__mach_task_self (), &data.addr, len, 1);
	  if (err)
	    {
	      __set_errno (err);
	      return -1;
	    }
	  dealloc = 1;
	}
      else
	data.ptr = data_buf;

      /* Copy the data into DATA.  */
      to_copy = len;
      buf = data.ptr;
      for (i = 0; i < len; i++)
	{
#define	min(a, b)	((a) > (b) ? (b) : (a))
	  size_t copy = min (message->msg_iov[i].iov_len, to_copy);

	  buf = __mempcpy (buf, message->msg_iov[i].iov_base, copy);

	  to_copy -= copy;
	  if (to_copy == 0)
	    break;
	}
    }

  /* Allocate enough room for ports.  */
  cmsg = CMSG_FIRSTHDR (message);
  for (; cmsg; cmsg = CMSG_NXTHDR ((struct msghdr *) message, cmsg))
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
      nports += (cmsg->cmsg_len - CMSG_ALIGN (sizeof (struct cmsghdr)))
		/ sizeof (int);

  if (nports)
    ports = __alloca (nports * sizeof (mach_port_t));

  nports = 0;
  for (cmsg = CMSG_FIRSTHDR (message);
       cmsg;
       cmsg = CMSG_NXTHDR ((struct msghdr *) message, cmsg))
    {
      if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS)
	{
	  /* SCM_RIGHTS support: send FDs.   */
	  fds = (int *) CMSG_DATA (cmsg);
	  nfds = (cmsg->cmsg_len - CMSG_ALIGN (sizeof (struct cmsghdr)))
		 / sizeof (int);

	  for (i = 0; i < nfds; i++)
	    {
	      err = HURD_DPORT_USE
		(fds[i],
		 ({
		   err = __io_restrict_auth (port, &ports[nports],
					     0, 0, 0, 0);
		   if (! err)
		     nports++;
		   /* We pass the flags in the control data.  */
		   fds[i] = descriptor->flags;
		   err;
		 }));

	      if (err)
		goto out;
	    }
	}
    }

  if (addr)
    {
      if (addr->sun_family == AF_LOCAL)
	{
	  char *name = _hurd_sun_path_dupa (addr, addr_len);
	  /* For the local domain, we must look up the name as a file
	     and talk to it with the ifsock protocol.  */
	  file_t file = __file_name_lookup (name, 0, 0);
	  if (file == MACH_PORT_NULL)
	    {
	      err = errno;
	      goto out;
	    }
	  err = __ifsock_getsockaddr (file, &aport);
	  __mach_port_deallocate (__mach_task_self (), file);
	  if (err == MIG_BAD_ID || err == EOPNOTSUPP)
	    /* The file did not grok the ifsock protocol.  */
	    err = ENOTSOCK;
	  if (err)
	    goto out;
	}
      else
	err = EIEIO;
    }

  err = HURD_DPORT_USE_CANCEL (fd,
			({
			  if (err)
			    err = __socket_create_address (port,
							   addr->sun_family,
							   (char *) addr,
							   addr_len,
							   &aport);
			  if (! err)
			    {
			      /* Send the data.  */
			      int cancel_oldtype = LIBC_CANCEL_ASYNC();
			      err = __socket_send (port, aport,
						   flags, data.ptr, len,
						   ports,
						   MACH_MSG_TYPE_COPY_SEND,
						   nports,
						   message->msg_control,
						   message->msg_controllen,
						   &amount);
			      LIBC_CANCEL_RESET (cancel_oldtype);
			      __mach_port_deallocate (__mach_task_self (),
						      aport);
			    }
			  err;
			}));
  socketrpc = 1;

 out:
  for (i = 0; i < nports; i++)
    __mach_port_deallocate (__mach_task_self (), ports[i]);

  if (dealloc)
    __vm_deallocate (__mach_task_self (), data.addr, len);

  if (socketrpc)
    return err ? __hurd_sockfail (fd, flags, err) : amount;
  else
    return __hurd_fail (err);
}

weak_alias (__libc_sendmsg, sendmsg)
weak_alias (__libc_sendmsg, __sendmsg)
