/*
 * pmap_rmt.c
 * Client interface to pmap rpc service.
 * remote call and broadcast service
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

#include <unistd.h>
#include <string.h>
#include <libintl.h>
#include <rpc/rpc.h>
#include <rpc/pmap_prot.h>
#include <rpc/pmap_clnt.h>
#include <rpc/pmap_rmt.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <sys/param.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <shlib-compat.h>

#define MAX_BROADCAST_SIZE 1400

extern u_long _create_xid (void);

static const struct timeval timeout = {3, 0};

/*
 * pmapper remote-call-service interface.
 * This routine is used to call the pmapper remote call service
 * which will look up a service program in the port maps, and then
 * remotely call that routine with the given parameters.  This allows
 * programs to do a lookup and call in one step.
 */
enum clnt_stat
pmap_rmtcall (struct sockaddr_in *addr, u_long prog, u_long vers, u_long proc,
	      xdrproc_t xdrargs, caddr_t argsp, xdrproc_t xdrres, caddr_t resp,
	      struct timeval tout, u_long *port_ptr)
{
  int socket = -1;
  CLIENT *client;
  struct rmtcallargs a;
  struct rmtcallres r;
  enum clnt_stat stat;

  addr->sin_port = htons (PMAPPORT);
  client = clntudp_create (addr, PMAPPROG, PMAPVERS, timeout, &socket);
  if (client != (CLIENT *) NULL)
    {
      a.prog = prog;
      a.vers = vers;
      a.proc = proc;
      a.args_ptr = argsp;
      a.xdr_args = xdrargs;
      r.port_ptr = port_ptr;
      r.results_ptr = resp;
      r.xdr_results = xdrres;
      stat = CLNT_CALL (client, PMAPPROC_CALLIT,
			(xdrproc_t)xdr_rmtcall_args,
			(caddr_t)&a, (xdrproc_t)xdr_rmtcallres,
			(caddr_t)&r, tout);
      CLNT_DESTROY (client);
    }
  else
    {
      stat = RPC_FAILED;
    }
  /* (void)__close(socket); CLNT_DESTROY already closed it */
  addr->sin_port = 0;
  return stat;
}
libc_hidden_nolink_sunrpc (pmap_rmtcall, GLIBC_2_0)


/*
 * XDR remote call arguments
 * written for XDR_ENCODE direction only
 */
bool_t
xdr_rmtcall_args (XDR *xdrs, struct rmtcallargs *cap)
{
  u_int lenposition, argposition, position;

  if (xdr_u_long (xdrs, &(cap->prog)) &&
      xdr_u_long (xdrs, &(cap->vers)) &&
      xdr_u_long (xdrs, &(cap->proc)))
    {
      u_long dummy_arglen = 0;
      lenposition = XDR_GETPOS (xdrs);
      if (!xdr_u_long (xdrs, &dummy_arglen))
	return FALSE;
      argposition = XDR_GETPOS (xdrs);
      if (!(*(cap->xdr_args)) (xdrs, cap->args_ptr))
	return FALSE;
      position = XDR_GETPOS (xdrs);
      cap->arglen = (u_long) position - (u_long) argposition;
      XDR_SETPOS (xdrs, lenposition);
      if (!xdr_u_long (xdrs, &(cap->arglen)))
	return FALSE;
      XDR_SETPOS (xdrs, position);
      return TRUE;
    }
  return FALSE;
}
libc_hidden_nolink_sunrpc (xdr_rmtcall_args, GLIBC_2_0)

/*
 * XDR remote call results
 * written for XDR_DECODE direction only
 */
bool_t
xdr_rmtcallres (XDR *xdrs, struct rmtcallres *crp)
{
  caddr_t port_ptr;

  port_ptr = (caddr_t) crp->port_ptr;
  if (xdr_reference (xdrs, &port_ptr, sizeof (u_long),
		     (xdrproc_t) xdr_u_long)
      && xdr_u_long (xdrs, &crp->resultslen))
    {
      crp->port_ptr = (u_long *) port_ptr;
      return (*(crp->xdr_results)) (xdrs, crp->results_ptr);
    }
  return FALSE;
}
libc_hidden_nolink_sunrpc (xdr_rmtcallres, GLIBC_2_0)


/*
 * The following is kludged-up support for simple rpc broadcasts.
 * Someday a large, complicated system will replace these trivial
 * routines which only support udp/ip .
 */

static int
getbroadcastnets (struct in_addr *addrs, int naddrs)
{
  struct ifaddrs *ifa;

  if (getifaddrs (&ifa) != 0)
    {
      perror ("broadcast: getifaddrs");
      return 0;
    }

  int i = 0;
  struct ifaddrs *run = ifa;
  while (run != NULL && i < naddrs)
    {
      if ((run->ifa_flags & IFF_BROADCAST) != 0
	  && (run->ifa_flags & IFF_UP) != 0
	  && run->ifa_addr != NULL
	  && run->ifa_addr->sa_family == AF_INET)
	/* Copy the broadcast address.  */
	addrs[i++] = ((struct sockaddr_in *) run->ifa_broadaddr)->sin_addr;

      run = run->ifa_next;
    }

  freeifaddrs (ifa);

  return i;
}


enum clnt_stat
clnt_broadcast (/* program number */
		u_long prog,
		/* version number */
		u_long vers,
		/* procedure number */
		u_long proc,
		/* xdr routine for args */
		xdrproc_t xargs,
		/* pointer to args */
		caddr_t argsp,
		/* xdr routine for results */
		xdrproc_t xresults,
		/* pointer to results */
		caddr_t resultsp,
		/* call with each result obtained */
		resultproc_t eachresult)
{
  enum clnt_stat stat = RPC_FAILED;
  AUTH *unix_auth = authunix_create_default ();
  XDR xdr_stream;
  XDR *xdrs = &xdr_stream;
  struct timeval t;
  int outlen, inlen, nets;
  socklen_t fromlen;
  int sock;
  int on = 1;
  struct pollfd fd;
  int milliseconds;
  int i;
  bool_t done = FALSE;
  u_long xid;
  u_long port;
  struct in_addr addrs[20];
  struct sockaddr_in baddr, raddr;	/* broadcast and response addresses */
  struct rmtcallargs a;
  struct rmtcallres r;
  struct rpc_msg msg;
  char outbuf[MAX_BROADCAST_SIZE], inbuf[UDPMSGSIZE];

  /*
   * initialization: create a socket, a broadcast address, and
   * preserialize the arguments into a send buffer.
   */
  if ((sock = __socket (AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
      perror (_("Cannot create socket for broadcast rpc"));
      stat = RPC_CANTSEND;
      goto done_broad;
    }
#ifdef SO_BROADCAST
  if (__setsockopt (sock, SOL_SOCKET, SO_BROADCAST, &on, sizeof (on)) < 0)
    {
      perror (_("Cannot set socket option SO_BROADCAST"));
      stat = RPC_CANTSEND;
      goto done_broad;
    }
#endif /* def SO_BROADCAST */
  fd.fd = sock;
  fd.events = POLLIN;
  nets = getbroadcastnets (addrs, sizeof (addrs) / sizeof (addrs[0]));
  memset ((char *) &baddr, 0, sizeof (baddr));
  baddr.sin_family = AF_INET;
  baddr.sin_port = htons (PMAPPORT);
  baddr.sin_addr.s_addr = htonl (INADDR_ANY);
/*      baddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY); */
  msg.rm_xid = xid = _create_xid ();
  t.tv_usec = 0;
  msg.rm_direction = CALL;
  msg.rm_call.cb_rpcvers = RPC_MSG_VERSION;
  msg.rm_call.cb_prog = PMAPPROG;
  msg.rm_call.cb_vers = PMAPVERS;
  msg.rm_call.cb_proc = PMAPPROC_CALLIT;
  msg.rm_call.cb_cred = unix_auth->ah_cred;
  msg.rm_call.cb_verf = unix_auth->ah_verf;
  a.prog = prog;
  a.vers = vers;
  a.proc = proc;
  a.xdr_args = xargs;
  a.args_ptr = argsp;
  r.port_ptr = &port;
  r.xdr_results = xresults;
  r.results_ptr = resultsp;
  xdrmem_create (xdrs, outbuf, MAX_BROADCAST_SIZE, XDR_ENCODE);
  if ((!xdr_callmsg (xdrs, &msg))
      || (!xdr_rmtcall_args (xdrs, &a)))
    {
      stat = RPC_CANTENCODEARGS;
      goto done_broad;
    }
  outlen = (int) xdr_getpos (xdrs);
  xdr_destroy (xdrs);
  /*
   * Basic loop: broadcast a packet and wait a while for response(s).
   * The response timeout grows larger per iteration.
   */
  for (t.tv_sec = 4; t.tv_sec <= 14; t.tv_sec += 2)
    {
      for (i = 0; i < nets; i++)
	{
	  baddr.sin_addr = addrs[i];
	  if (__sendto (sock, outbuf, outlen, 0,
			(struct sockaddr *) &baddr,
			sizeof (struct sockaddr)) != outlen)
	    {
	      perror (_("Cannot send broadcast packet"));
	      stat = RPC_CANTSEND;
	      goto done_broad;
	    }
	}
      if (eachresult == NULL)
	{
	  stat = RPC_SUCCESS;
	  goto done_broad;
	}
    recv_again:
      msg.acpted_rply.ar_verf = _null_auth;
      msg.acpted_rply.ar_results.where = (caddr_t) & r;
      msg.acpted_rply.ar_results.proc = (xdrproc_t) xdr_rmtcallres;
      milliseconds = t.tv_sec * 1000 + t.tv_usec / 1000;
      switch (__poll(&fd, 1, milliseconds))
	{

	case 0:		/* timed out */
	  stat = RPC_TIMEDOUT;
	  continue;

	case -1:		/* some kind of error */
	  if (errno == EINTR)
	    goto recv_again;
	  perror (_("Broadcast poll problem"));
	  stat = RPC_CANTRECV;
	  goto done_broad;

	}			/* end of poll results switch */
    try_again:
      fromlen = sizeof (struct sockaddr);
      inlen = __recvfrom (sock, inbuf, UDPMSGSIZE, 0,
			  (struct sockaddr *) &raddr, &fromlen);
      if (inlen < 0)
	{
	  if (errno == EINTR)
	    goto try_again;
	  perror (_("Cannot receive reply to broadcast"));
	  stat = RPC_CANTRECV;
	  goto done_broad;
	}
      if ((size_t) inlen < sizeof (u_long))
	goto recv_again;
      /*
       * see if reply transaction id matches sent id.
       * If so, decode the results.
       */
      xdrmem_create (xdrs, inbuf, (u_int) inlen, XDR_DECODE);
      if (xdr_replymsg (xdrs, &msg))
	{
	  if (((uint32_t) msg.rm_xid == (uint32_t) xid) &&
	      (msg.rm_reply.rp_stat == MSG_ACCEPTED) &&
	      (msg.acpted_rply.ar_stat == SUCCESS))
	    {
	      raddr.sin_port = htons ((u_short) port);
	      done = (*eachresult) (resultsp, &raddr);
	    }
	  /* otherwise, we just ignore the errors ... */
	}
      else
	{
#ifdef notdef
	  /* some kind of deserialization problem ... */
	  if ((uint32_t) msg.rm_xid == (uint32_t) xid)
	    fprintf (stderr, "Broadcast deserialization problem");
	  /* otherwise, just random garbage */
#endif
	}
      xdrs->x_op = XDR_FREE;
      msg.acpted_rply.ar_results.proc = (xdrproc_t)xdr_void;
      (void) xdr_replymsg (xdrs, &msg);
      (void) (*xresults) (xdrs, resultsp);
      xdr_destroy (xdrs);
      if (done)
	{
	  stat = RPC_SUCCESS;
	  goto done_broad;
	}
      else
	{
	  goto recv_again;
	}
    }
done_broad:
  (void) __close (sock);
  AUTH_DESTROY (unix_auth);
  return stat;
}
libc_hidden_nolink_sunrpc (clnt_broadcast, GLIBC_2_0)
