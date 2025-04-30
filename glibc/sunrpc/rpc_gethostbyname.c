/* IPv4-only variant of gethostbyname.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <netdb.h>
#include <rpc/rpc.h>
#include <scratch_buffer.h>
#include <string.h>

int
__libc_rpc_gethostbyname (const char *host, struct sockaddr_in *addr)
{
  struct hostent hostbuf;
  struct hostent *hp = NULL;
  int herr;
  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);

  while (__gethostbyname2_r (host, AF_INET,
                             &hostbuf, tmpbuf.data, tmpbuf.length, &hp,
                             &herr) != 0
         || hp == NULL)
    if (herr != NETDB_INTERNAL || errno != ERANGE)
      {
        struct rpc_createerr *ce = &get_rpc_createerr ();
        ce->cf_stat = RPC_UNKNOWNHOST;
        scratch_buffer_free (&tmpbuf);
        return -1;
      }
    else
      {
        if (!scratch_buffer_grow (&tmpbuf))
          {
            /* If memory allocation failed, allocating the RPC error
               structure might could as well, so this could lead to a
               crash.  */
            struct rpc_createerr *ce = &get_rpc_createerr ();
            ce->cf_stat = RPC_SYSTEMERROR;
            ce->cf_error.re_errno = ENOMEM;
            return -1;
          }
      }

  if (hp->h_addrtype != AF_INET || hp->h_length != sizeof (addr->sin_addr))
    {
      struct rpc_createerr *ce = &get_rpc_createerr ();
      ce->cf_stat = RPC_SYSTEMERROR;
      ce->cf_error.re_errno = EAFNOSUPPORT;
      scratch_buffer_free (&tmpbuf);
      return -1;
    }

  addr->sin_family = AF_INET;
  addr->sin_port = htons (0);
  memcpy (&addr->sin_addr, hp->h_addr, sizeof (addr->sin_addr));
  scratch_buffer_free (&tmpbuf);
  return 0;
}
