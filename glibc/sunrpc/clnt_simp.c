/*
 * clnt_simple.c
 * Simplified front end to rpc.
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

#include <alloca.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <rpc/rpc.h>
#include <sys/socket.h>
#include <netdb.h>
#include <string.h>
#include <shlib-compat.h>

struct callrpc_private_s
  {
    CLIENT *client;
    int socket;
    u_long oldprognum, oldversnum, valid;
    char *oldhost;
  };
#define callrpc_private RPC_THREAD_VARIABLE(callrpc_private_s)

int
callrpc (const char *host, u_long prognum, u_long versnum, u_long procnum,
	 xdrproc_t inproc, const char *in, xdrproc_t outproc, char *out)
{
  struct callrpc_private_s *crp = callrpc_private;
  struct sockaddr_in server_addr;
  enum clnt_stat clnt_stat;
  struct timeval timeout, tottimeout;

  if (crp == 0)
    {
      crp = (struct callrpc_private_s *) calloc (1, sizeof (*crp));
      if (crp == 0)
	return 0;
      callrpc_private = crp;
    }
  if (crp->oldhost == NULL)
    {
      crp->oldhost = malloc (256);
      crp->oldhost[0] = 0;
      crp->socket = RPC_ANYSOCK;
    }
  if (crp->valid && crp->oldprognum == prognum && crp->oldversnum == versnum
      && strcmp (crp->oldhost, host) == 0)
    {
      /* reuse old client */
    }
  else
    {
      crp->valid = 0;
      if (crp->socket != RPC_ANYSOCK)
	{
	  (void) __close (crp->socket);
	  crp->socket = RPC_ANYSOCK;
	}
      if (crp->client)
	{
	  clnt_destroy (crp->client);
	  crp->client = NULL;
	}

      if (__libc_rpc_gethostbyname (host, &server_addr) != 0)
	return (int) get_rpc_createerr().cf_stat;

      timeout.tv_usec = 0;
      timeout.tv_sec = 5;
      if ((crp->client = clntudp_create (&server_addr, (u_long) prognum,
			  (u_long) versnum, timeout, &crp->socket)) == NULL)
	return (int) get_rpc_createerr().cf_stat;
      crp->valid = 1;
      crp->oldprognum = prognum;
      crp->oldversnum = versnum;
      (void) strncpy (crp->oldhost, host, 255);
      crp->oldhost[255] = '\0';
    }
  tottimeout.tv_sec = 25;
  tottimeout.tv_usec = 0;
  clnt_stat = clnt_call (crp->client, procnum, inproc, (char *) in,
			 outproc, out, tottimeout);
  /*
   * if call failed, empty cache
   */
  if (clnt_stat != RPC_SUCCESS)
    crp->valid = 0;
  return (int) clnt_stat;
}
libc_hidden_nolink_sunrpc (callrpc, GLIBC_2_0)

void
__rpc_thread_clnt_cleanup (void)
{
	struct callrpc_private_s *rcp = RPC_THREAD_VARIABLE(callrpc_private_s);

	if (rcp) {
		if (rcp->client)
			CLNT_DESTROY (rcp->client);
		free (rcp);
	}
}
