/*
 * pmap_getport.c
 * Client interface to pmap rpc service.
 *
 * Copyright (c) 2010, Oracle America, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the "Oracle America, Inc." nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdbool.h>
#include <unistd.h>
#include <rpc/rpc.h>
#include <rpc/pmap_prot.h>
#include <rpc/pmap_clnt.h>
#include <sys/socket.h>
#include <shlib-compat.h>

/*
 * Create a socket that is locally bound to a non-reserve port. For
 * any failures, -1 is returned which will cause the RPC code to
 * create the socket.
 */
int
__get_socket (struct sockaddr_in *saddr)
{
  int so = __socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (so < 0)
    return -1;

  struct sockaddr_in laddr;
  socklen_t namelen = sizeof (laddr);
  laddr.sin_family = AF_INET;
  laddr.sin_port = 0;
  laddr.sin_addr.s_addr = htonl (INADDR_ANY);

  int cc = __bind (so, (struct sockaddr *) &laddr, namelen);
  if (__glibc_unlikely (cc < 0))
    {
    fail:
      __close (so);
      return -1;
    }

  cc = __connect (so, (struct sockaddr *) saddr, namelen);
  if (__glibc_unlikely (cc < 0))
    goto fail;

  return so;
}


/*
 * Find the mapped port for program,version.
 * Internal version with additional parameters.
 * Calls the pmap service remotely to do the lookup.
 * Returns 0 if no map exists.
 */
u_short
__libc_rpc_getport (struct sockaddr_in *address, u_long program,
		    u_long version, u_int protocol, time_t timeout_sec,
		    time_t tottimeout_sec)
{
  const struct timeval timeout = {timeout_sec, 0};
  const struct timeval tottimeout = {tottimeout_sec, 0};

  u_short port = 0;
  int socket = -1;
  CLIENT *client;
  struct pmap parms;
  bool closeit = false;

  address->sin_port = htons (PMAPPORT);
  if (protocol == IPPROTO_TCP)
    {
      /* Don't need a reserved port to get ports from the portmapper.  */
      socket = __get_socket(address);
      if (socket != -1)
	closeit = true;
      client = clnttcp_create (address, PMAPPROG, PMAPVERS, &socket,
			       RPCSMALLMSGSIZE, RPCSMALLMSGSIZE);
    }
  else
    client = clntudp_bufcreate (address, PMAPPROG, PMAPVERS, timeout,
				&socket, RPCSMALLMSGSIZE, RPCSMALLMSGSIZE);
  if (client != (CLIENT *) NULL)
    {
      struct rpc_createerr *ce = &get_rpc_createerr ();
      parms.pm_prog = program;
      parms.pm_vers = version;
      parms.pm_prot = protocol;
      parms.pm_port = 0;	/* not needed or used */
      if (CLNT_CALL (client, PMAPPROC_GETPORT, (xdrproc_t)xdr_pmap,
		     (caddr_t)&parms, (xdrproc_t)xdr_u_short,
		     (caddr_t)&port, tottimeout) != RPC_SUCCESS)
	{
	  ce->cf_stat = RPC_PMAPFAILURE;
	  clnt_geterr (client, &ce->cf_error);
	}
      else if (port == 0)
	{
	  ce->cf_stat = RPC_PROGNOTREGISTERED;
	}
      CLNT_DESTROY (client);
    }
  /* We only need to close the socket here if we opened  it.  */
  if (closeit)
    (void) __close (socket);
  address->sin_port = 0;
  return port;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (__libc_rpc_getport)
#else
libc_hidden_nolink_sunrpc (__libc_rpc_getport, GLIBC_PRIVATE)
#endif


/*
 * Find the mapped port for program,version.
 * Calls the pmap service remotely to do the lookup.
 * Returns 0 if no map exists.
 */
u_short
pmap_getport (struct sockaddr_in *address, u_long program, u_long version,
	      u_int protocol)
{
  return __libc_rpc_getport (address, program, version, protocol, 5, 60);
}
libc_hidden_nolink_sunrpc (pmap_getport, GLIBC_2_0)
