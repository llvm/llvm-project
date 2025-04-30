/*
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
/*
 * pmap_clnt.c
 * Client interface to pmap rpc service.
 */

#include <stdio.h>
#include <unistd.h>
#include <libintl.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <rpc/rpc.h>
#include <rpc/pmap_prot.h>
#include <rpc/pmap_clnt.h>
#include <shlib-compat.h>

/*
 * Same as get_myaddress, but we try to use the loopback
 * interface. portmap caches interfaces, and on DHCP clients,
 * it could be that only loopback is started at this time.
 */
static bool_t
__get_myaddress (struct sockaddr_in *addr)
{
  struct ifaddrs *ifa;

  if (getifaddrs (&ifa) != 0)
    {
      perror ("get_myaddress: getifaddrs");
      exit (1);
    }

  int loopback = 1;
  struct ifaddrs *run;

 again:
  run = ifa;
  while (run != NULL)
    {
      if ((run->ifa_flags & IFF_UP)
	  && run->ifa_addr != NULL
	  && run->ifa_addr->sa_family == AF_INET
	  && ((run->ifa_flags & IFF_LOOPBACK) || loopback == 0))
	{
	  *addr = *((struct sockaddr_in *) run->ifa_addr);
	  addr->sin_port = htons (PMAPPORT);
	  goto out;
	}

      run = run->ifa_next;
    }

  if (loopback == 1)
    {
      loopback = 0;
      goto again;
    }
 out:
  freeifaddrs (ifa);

  return run == NULL ? FALSE : TRUE;
}


static const struct timeval timeout = {5, 0};
static const struct timeval tottimeout = {60, 0};

/*
 * Set a mapping between program,version and port.
 * Calls the pmap service remotely to do the mapping.
 */
bool_t
pmap_set (u_long program, u_long version, int protocol, u_short port)
{
  struct sockaddr_in myaddress;
  int socket = -1;
  CLIENT *client;
  struct pmap parms;
  bool_t rslt;

  if (!__get_myaddress (&myaddress))
    return FALSE;
  client = clntudp_bufcreate (&myaddress, PMAPPROG, PMAPVERS, timeout, &socket,
			      RPCSMALLMSGSIZE, RPCSMALLMSGSIZE);
  if (client == (CLIENT *) NULL)
    return (FALSE);
  parms.pm_prog = program;
  parms.pm_vers = version;
  parms.pm_prot = protocol;
  parms.pm_port = port;
  if (CLNT_CALL (client, PMAPPROC_SET, (xdrproc_t)xdr_pmap,
		 (caddr_t)&parms, (xdrproc_t)xdr_bool, (caddr_t)&rslt,
		 tottimeout) != RPC_SUCCESS)
    {
      clnt_perror (client, _("Cannot register service"));
      rslt = FALSE;
    }
  CLNT_DESTROY (client);
  /* (void)close(socket); CLNT_DESTROY closes it */
  return rslt;
}
libc_hidden_nolink_sunrpc (pmap_set, GLIBC_2_0)

/*
 * Remove the mapping between program,version and port.
 * Calls the pmap service remotely to do the un-mapping.
 */
bool_t
pmap_unset (u_long program, u_long version)
{
  struct sockaddr_in myaddress;
  int socket = -1;
  CLIENT *client;
  struct pmap parms;
  bool_t rslt;

  if (!__get_myaddress (&myaddress))
    return FALSE;
  client = clntudp_bufcreate (&myaddress, PMAPPROG, PMAPVERS, timeout, &socket,
			      RPCSMALLMSGSIZE, RPCSMALLMSGSIZE);
  if (client == (CLIENT *) NULL)
    return FALSE;
  parms.pm_prog = program;
  parms.pm_vers = version;
  parms.pm_port = parms.pm_prot = 0;
  CLNT_CALL (client, PMAPPROC_UNSET, (xdrproc_t)xdr_pmap,
	     (caddr_t)&parms, (xdrproc_t)xdr_bool, (caddr_t)&rslt,
	     tottimeout);
  CLNT_DESTROY (client);
  /* (void)close(socket); CLNT_DESTROY already closed it */
  return rslt;
}
libc_hidden_nolink_sunrpc (pmap_unset, GLIBC_2_0)
