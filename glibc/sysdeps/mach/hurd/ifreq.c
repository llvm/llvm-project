/* Fetch the host's network interface list.  Hurd version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <ifreq.h>
#include <hurd.h>
#include <hurd/pfinet.h>
#include <sys/mman.h>


void
__ifreq (struct ifreq **ifreqs, int *num_ifs, int sockfd)
{
  file_t server;

  server = _hurd_socket_server (PF_INET, 0);
  if (server == MACH_PORT_NULL)
    {
    out:
      *num_ifs = 0;
      *ifreqs = NULL;
    }
  else
    {
      char *data = NULL;
      size_t len = 0;
      error_t err = __pfinet_siocgifconf (server, -1, &data, &len);
      if (err == MACH_SEND_INVALID_DEST || err == MIG_SERVER_DIED)
	{
	  /* On the first use of the socket server during the operation,
	     allow for the old server port dying.  */
	  server = _hurd_socket_server (PF_INET, 1);
	  if (server == MACH_PORT_NULL)
	    goto out;
	  err = __pfinet_siocgifconf (server, -1, (data_t *) ifreqs, &len);
	}
      if (err)
	goto out;

      if (len % sizeof (struct ifreq) != 0)
	{
	  __munmap (data, len);
	  errno = EGRATUITOUS;
	  goto out;
	}
      *num_ifs = len / sizeof (struct ifreq);
      *ifreqs = (struct ifreq *) data;
    }

}
