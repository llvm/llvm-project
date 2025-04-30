/*
 * pmap_getmap.c
 * Client interface to pmap rpc service.
 * contains pmap_getmaps, which is only tcp service involved
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

#include <rpc/rpc.h>
#include <rpc/pmap_prot.h>
#include <rpc/pmap_clnt.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>
#include <libintl.h>
#include <unistd.h>
#include <not-cancel.h>
#include <shlib-compat.h>


/*
 * Get a copy of the current port maps.
 * Calls the pmap service remotely to do get the maps.
 */
struct pmaplist *
pmap_getmaps (struct sockaddr_in *address)
{
  struct pmaplist *head = (struct pmaplist *) NULL;
  struct timeval minutetimeout;
  CLIENT *client;
  bool closeit = false;

  minutetimeout.tv_sec = 60;
  minutetimeout.tv_usec = 0;
  address->sin_port = htons (PMAPPORT);

  /* Don't need a reserved port to get ports from the portmapper.  */
  int socket = __get_socket (address);
  if (socket != -1)
    closeit = true;

  client = clnttcp_create (address, PMAPPROG, PMAPVERS, &socket, 50, 500);
  if (client != (CLIENT *) NULL)
    {
      if (CLNT_CALL (client, PMAPPROC_DUMP, (xdrproc_t)xdr_void, NULL,
		     (xdrproc_t)xdr_pmaplist, (caddr_t)&head,
		     minutetimeout) != RPC_SUCCESS)
	{
	  clnt_perror (client, _("pmap_getmaps.c: rpc problem"));
	}
      CLNT_DESTROY (client);
    }
  /* We only need to close the socket here if we opened  it.  */
  if (closeit)
    __close_nocancel (socket);
  address->sin_port = 0;
  return head;
}
libc_hidden_nolink_sunrpc (pmap_getmaps, GLIBC_2_0)
