/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1997.

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

#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <rpc/pmap_prot.h>
#include <rpc/pmap_clnt.h>
#include <rpcsvc/nis.h>

#include "nis_intern.h"

/* Private data kept per client handle, from sunrpc/clnt_udp.c */
struct cu_data
  {
    int cu_sock;
    bool_t cu_closeit;
    struct sockaddr_in cu_raddr;
    int cu_rlen;
    struct timeval cu_wait;
    struct timeval cu_total;
    struct rpc_err cu_error;
    XDR cu_outxdrs;
    u_int cu_xdrpos;
    u_int cu_sendsz;
    char *cu_outbuf;
    u_int cu_recvsz;
    char cu_inbuf[1];
  };


/*
 * Find the mapped port for program,version.
 * Calls the pmap service remotely to do the lookup.
 * Returns 0 if no map exists.
 */
u_short
__pmap_getnisport (struct sockaddr_in *address, u_long program,
		   u_long version, u_int protocol)
{
  return __libc_rpc_getport (address, program, version, protocol, 1, 1);
}

/* This is now the public function, which should find the fastest server */

struct findserv_req
{
  struct sockaddr_in sin;
  uint32_t xid;
  u_int server_nr;
  u_int server_ep;
};


static long int
__nis_findfastest_with_timeout (dir_binding *bind,
				const struct timeval *timeout)
{
  static const struct timeval TIMEOUT00 = { 0, 0 };
  struct findserv_req *pings;
  struct sockaddr_in sin, saved_sin;
  int found = -1;
  uint32_t xid_seed;
  int sock, dontblock = 1;
  CLIENT *clnt;
  u_long i, j, pings_count, pings_max, fastest = -1;
  struct cu_data *cu;

  pings_max = bind->server_len * 2;	/* Reserve a little bit more memory
					   for multihomed hosts */
  pings_count = 0;
  pings = malloc (sizeof (struct findserv_req) * pings_max);
  xid_seed = (uint32_t) (time (NULL) ^ getpid ());

  if (__glibc_unlikely (pings == NULL))
    return -1;

  memset (&sin, '\0', sizeof (sin));
  sin.sin_family = AF_INET;
  for (i = 0; i < bind->server_len; i++)
    for (j = 0; j < bind->server_val[i].ep.ep_len; ++j)
      if (strcmp (bind->server_val[i].ep.ep_val[j].family, "inet") == 0)
	if ((bind->server_val[i].ep.ep_val[j].proto == NULL)
	    || (bind->server_val[i].ep.ep_val[j].proto[0] == '-')
	    || (bind->server_val[i].ep.ep_val[j].proto[0] == '\0'))
	  {
	    sin.sin_addr.s_addr =
	      inetstr2int (bind->server_val[i].ep.ep_val[j].uaddr);
	    if (sin.sin_addr.s_addr == 0)
	      continue;
	    sin.sin_port = htons (__pmap_getnisport (&sin, NIS_PROG,
						     NIS_VERSION,
						     IPPROTO_UDP));
	    if (sin.sin_port == 0)
	      continue;

	    if (pings_count >= pings_max)
	      {
		struct findserv_req *new_pings;

		pings_max += 10;
		new_pings = realloc (pings, sizeof (struct findserv_req) *
				     pings_max);
		if (__glibc_unlikely (new_pings == NULL))
		  {
		    free (pings);
		    return -1;
		  }
		pings = new_pings;
	      }
	    memcpy ((char *) &pings[pings_count].sin, (char *) &sin,
		    sizeof (sin));
	    memcpy ((char *)&saved_sin, (char *)&sin, sizeof (sin));
	    pings[pings_count].xid = xid_seed + pings_count;
	    pings[pings_count].server_nr = i;
	    pings[pings_count].server_ep = j;
	    ++pings_count;
	  }

  /* Make sure at least one server was assigned */
  if (pings_count == 0)
    {
      free (pings);
      return -1;
    }

  /* Create RPC handle */
  sock = socket (AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);
  clnt = clntudp_create (&saved_sin, NIS_PROG, NIS_VERSION, *timeout, &sock);
  if (clnt == NULL)
    {
      close (sock);
      free (pings);
      return -1;
    }
  auth_destroy (clnt->cl_auth);
  clnt->cl_auth = authunix_create_default ();
  cu = (struct cu_data *) clnt->cl_private;
  ioctl (sock, FIONBIO, &dontblock);
  /* Send to all servers the NULLPROC */
  for (i = 0; i < pings_count; ++i)
    {
      /* clntudp_call() will increment, subtract one */
      *((uint32_t *) (cu->cu_outbuf)) = pings[i].xid - 1;
      memcpy ((char *) &cu->cu_raddr, (char *) &pings[i].sin,
	      sizeof (struct sockaddr_in));
      /* Transmit to NULLPROC, return immediately. */
      clnt_call (clnt, NULLPROC,
		 (xdrproc_t) xdr_void, (caddr_t) 0,
		 (xdrproc_t) xdr_void, (caddr_t) 0, TIMEOUT00);
    }

  while (found == -1) {
    /* Receive reply from NULLPROC asynchronously. Note null inproc. */
    int rc = clnt_call (clnt, NULLPROC,
			(xdrproc_t) NULL, (caddr_t) 0,
			(xdrproc_t) xdr_void, (caddr_t) 0,
			*timeout);
    if (RPC_SUCCESS == rc) {
      uint32_t val;
      memcpy (&val, cu->cu_inbuf, sizeof (uint32_t));
      fastest = val - xid_seed;
      if (fastest < pings_count) {
	bind->server_used = pings[fastest].server_nr;
	bind->current_ep = pings[fastest].server_ep;
	found = 1;
      }
    } else {
      /*      clnt_perror(clnt, "__nis_findfastest"); */
      break;
    }
  }


  auth_destroy (clnt->cl_auth);
  clnt_destroy (clnt);
  close (sock);

  free (pings);

  return found;
}


long int
__nis_findfastest (dir_binding *bind)
{
  struct timeval timeout = { __NIS_PING_TIMEOUT_START, 0 };
  long int found = -1;
  long int retry = __NIS_PING_RETRY + 1;

  while (retry--)
    {
      found = __nis_findfastest_with_timeout (bind, &timeout);
      if (found != -1)
	break;
      timeout.tv_sec += __NIS_PING_TIMEOUT_INCREMENT;
    }

  return found;
}
