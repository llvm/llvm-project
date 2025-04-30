/*
 * clnt_udp.c, Implements a UDP/IP based, client side RPC.
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

#include <stdio.h>
#include <unistd.h>
#include <libintl.h>
#include <rpc/rpc.h>
#include <rpc/xdr.h>
#include <rpc/clnt.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netdb.h>
#include <errno.h>
#include <stdint.h>
#include <rpc/pmap_clnt.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <wchar.h>
#include <fcntl.h>

#ifdef IP_RECVERR
#include <errqueue.h>
#include <sys/uio.h>
#endif

#include <kernel-features.h>
#include <inet/net-internal.h>
#include <shlib-compat.h>
#include <libc-diag.h>

extern u_long _create_xid (void);

/*
 * UDP bases client side rpc operations
 */
static enum clnt_stat clntudp_call (CLIENT *, u_long, xdrproc_t, caddr_t,
				    xdrproc_t, caddr_t, struct timeval);
static void clntudp_abort (void);
static void clntudp_geterr (CLIENT *, struct rpc_err *);
static bool_t clntudp_freeres (CLIENT *, xdrproc_t, caddr_t);
static bool_t clntudp_control (CLIENT *, int, char *);
static void clntudp_destroy (CLIENT *);

static const struct clnt_ops udp_ops =
{
  clntudp_call,
  clntudp_abort,
  clntudp_geterr,
  clntudp_freeres,
  clntudp_destroy,
  clntudp_control
};

/*
 * Private data kept per client handle.  This private struct is
 * unfortunately part of the ABI; ypbind contains a copy of it and
 * accesses it through CLIENT::cl_private field.
 */
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
 * Create a UDP based client handle.
 * If *sockp<0, *sockp is set to a newly created UPD socket.
 * If raddr->sin_port is 0 a binder on the remote machine
 * is consulted for the correct port number.
 * NB: It is the clients responsibility to close *sockp.
 * NB: The rpch->cl_auth is initialized to null authentication.
 *     Caller may wish to set this something more useful.
 *
 * wait is the amount of time used between retransmitting a call if
 * no response has been heard; retransmission occurs until the actual
 * rpc call times out.
 *
 * sendsz and recvsz are the maximum allowable packet sizes that can be
 * sent and received.
 */
CLIENT *
__libc_clntudp_bufcreate (struct sockaddr_in *raddr, u_long program,
			  u_long version, struct timeval wait, int *sockp,
			  u_int sendsz, u_int recvsz, int flags)
{
  CLIENT *cl;
  struct cu_data *cu = NULL;
  struct rpc_msg call_msg;

  cl = (CLIENT *) mem_alloc (sizeof (CLIENT));
  sendsz = ((sendsz + 3) / 4) * 4;
  recvsz = ((recvsz + 3) / 4) * 4;
  cu = (struct cu_data *) mem_alloc (sizeof (*cu) + sendsz + recvsz);
  if (cl == NULL || cu == NULL)
    {
      struct rpc_createerr *ce = &get_rpc_createerr ();
      (void) __fxprintf (NULL, "%s: %s",
			 "clntudp_create", _("out of memory\n"));
      ce->cf_stat = RPC_SYSTEMERROR;
      ce->cf_error.re_errno = ENOMEM;
      goto fooy;
    }
  cu->cu_outbuf = &cu->cu_inbuf[recvsz];

  if (raddr->sin_port == 0)
    {
      u_short port;
      if ((port =
	   pmap_getport (raddr, program, version, IPPROTO_UDP)) == 0)
	{
	  goto fooy;
	}
      raddr->sin_port = htons (port);
    }
  cl->cl_ops = (struct clnt_ops *) &udp_ops;
  cl->cl_private = (caddr_t) cu;
  cu->cu_raddr = *raddr;
  cu->cu_rlen = sizeof (cu->cu_raddr);
  cu->cu_wait = wait;
  cu->cu_total.tv_sec = -1;
  cu->cu_total.tv_usec = -1;
  cu->cu_sendsz = sendsz;
  cu->cu_recvsz = recvsz;
  call_msg.rm_xid = _create_xid ();
  call_msg.rm_direction = CALL;
  call_msg.rm_call.cb_rpcvers = RPC_MSG_VERSION;
  call_msg.rm_call.cb_prog = program;
  call_msg.rm_call.cb_vers = version;
  xdrmem_create (&(cu->cu_outxdrs), cu->cu_outbuf, sendsz, XDR_ENCODE);
  if (!xdr_callhdr (&(cu->cu_outxdrs), &call_msg))
    {
      goto fooy;
    }
  cu->cu_xdrpos = XDR_GETPOS (&(cu->cu_outxdrs));
  if (*sockp < 0)
    {
      *sockp = __socket (AF_INET, SOCK_DGRAM|SOCK_NONBLOCK|flags, IPPROTO_UDP);
      if (__glibc_unlikely (*sockp < 0))
	{
	  struct rpc_createerr *ce = &get_rpc_createerr ();
	  ce->cf_stat = RPC_SYSTEMERROR;
	  ce->cf_error.re_errno = errno;
	  goto fooy;
	}
      /* attempt to bind to prov port */
      (void) bindresvport (*sockp, (struct sockaddr_in *) 0);
#ifdef IP_RECVERR
      {
	int on = 1;
	__setsockopt (*sockp, SOL_IP, IP_RECVERR, &on, sizeof(on));
      }
#endif
      cu->cu_closeit = TRUE;
    }
  else
    {
      cu->cu_closeit = FALSE;
    }
  cu->cu_sock = *sockp;
  cl->cl_auth = authnone_create ();
  return cl;
fooy:
  if (cu)
    mem_free ((caddr_t) cu, sizeof (*cu) + sendsz + recvsz);
  if (cl)
    mem_free ((caddr_t) cl, sizeof (CLIENT));
  return (CLIENT *) NULL;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (__libc_clntudp_bufcreate)
#else
libc_hidden_nolink_sunrpc (__libc_clntudp_bufcreate, GLIBC_PRIVATE)
#endif

CLIENT *
clntudp_bufcreate (struct sockaddr_in *raddr, u_long program, u_long version,
		   struct timeval wait, int *sockp, u_int sendsz,
		   u_int recvsz)
{
  return __libc_clntudp_bufcreate (raddr, program, version, wait,
				   sockp, sendsz, recvsz, 0);
}
libc_hidden_nolink_sunrpc (clntudp_bufcreate, GLIBC_2_0)

CLIENT *
clntudp_create (struct sockaddr_in *raddr, u_long program, u_long version,
		struct timeval wait, int *sockp)
{
  return __libc_clntudp_bufcreate (raddr, program, version, wait,
				   sockp, UDPMSGSIZE, UDPMSGSIZE, 0);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (clntudp_create)
#else
libc_hidden_nolink_sunrpc (clntudp_create, GLIBC_2_0)
#endif

static int
is_network_up (int sock)
{
  struct ifaddrs *ifa;

  if (getifaddrs (&ifa) != 0)
    return 0;

  struct ifaddrs *run = ifa;
  while (run != NULL)
    {
      if ((run->ifa_flags & IFF_UP) != 0
	  && run->ifa_addr != NULL
	  && run->ifa_addr->sa_family == AF_INET)
	break;

      run = run->ifa_next;
    }

  freeifaddrs (ifa);

  return run != NULL;
}

static enum clnt_stat
clntudp_call (/* client handle */
	      CLIENT *cl,
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
	      /* seconds to wait before giving up */
	      struct timeval utimeout)
{
  struct cu_data *cu = (struct cu_data *) cl->cl_private;
  XDR *xdrs;
  int outlen = 0;
  int inlen;
  socklen_t fromlen;
  struct pollfd fd;
  struct sockaddr_in from;
  struct rpc_msg reply_msg;
  XDR reply_xdrs;
  bool_t ok;
  int nrefreshes = 2;		/* number of times to refresh cred */
  int anyup;			/* any network interface up */

  struct deadline_current_time current_time = __deadline_current_time ();
  /* GCC 10 for MIPS reports total_deadline as possibly used
     uninitialized; see
     <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=91691>.  In fact it
     is initialized conditionally and only ever used under the same
     condition.  The same warning is also disabled in
     inet/net-internal.h because in some other configurations GCC
     gives the warning in an inline function.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (10, "-Wmaybe-uninitialized");
  struct deadline total_deadline; /* Determined once by overall timeout.  */
  DIAG_POP_NEEDS_COMMENT;
  struct deadline response_deadline; /* Determined anew for each query.  */

  /* Choose the timeout value.  For non-sending usage (xargs == NULL),
     the total deadline does not matter, only cu->cu_wait is used
     below.  */
  if (xargs != NULL)
    {
      struct timeval tv;
      if (cu->cu_total.tv_usec == -1)
	/* Use supplied timeout.  */
	tv = utimeout;
      else
	/* Use default timeout.  */
	tv = cu->cu_total;
      if (!__is_timeval_valid_timeout (tv))
	return (cu->cu_error.re_status = RPC_TIMEDOUT);
      total_deadline = __deadline_from_timeval (current_time, tv);
    }

  /* Guard against bad timeout specification.  */
  if (!__is_timeval_valid_timeout (cu->cu_wait))
    return (cu->cu_error.re_status = RPC_TIMEDOUT);

call_again:
  xdrs = &(cu->cu_outxdrs);
  if (xargs == NULL)
    goto get_reply;
  xdrs->x_op = XDR_ENCODE;
  XDR_SETPOS (xdrs, cu->cu_xdrpos);
  /*
   * the transaction is the first thing in the out buffer
   */
  (*(uint32_t *) (cu->cu_outbuf))++;
  if ((!XDR_PUTLONG (xdrs, (long *) &proc)) ||
      (!AUTH_MARSHALL (cl->cl_auth, xdrs)) ||
      (!(*xargs) (xdrs, argsp)))
    return (cu->cu_error.re_status = RPC_CANTENCODEARGS);
  outlen = (int) XDR_GETPOS (xdrs);

send_again:
  if (__sendto (cu->cu_sock, cu->cu_outbuf, outlen, 0,
		(struct sockaddr *) &(cu->cu_raddr), cu->cu_rlen)
      != outlen)
    {
      cu->cu_error.re_errno = errno;
      return (cu->cu_error.re_status = RPC_CANTSEND);
    }

  /* sendto may have blocked, so recompute the current time.  */
  current_time = __deadline_current_time ();
 get_reply:
  response_deadline = __deadline_from_timeval (current_time, cu->cu_wait);

  reply_msg.acpted_rply.ar_verf = _null_auth;
  reply_msg.acpted_rply.ar_results.where = resultsp;
  reply_msg.acpted_rply.ar_results.proc = xresults;
  fd.fd = cu->cu_sock;
  fd.events = POLLIN;
  anyup = 0;

  /* Per-response retry loop.  current_time must be up-to-date at the
     top of the loop.  */
  for (;;)
    {
      int milliseconds;
      if (xargs != NULL)
	{
	  if (__deadline_elapsed (current_time, total_deadline))
	    /* Overall timeout expired.  */
	    return (cu->cu_error.re_status = RPC_TIMEDOUT);
	  milliseconds = __deadline_to_ms
	    (current_time, __deadline_first (total_deadline,
					     response_deadline));
	  if (milliseconds == 0)
	    /* Per-query timeout expired.  */
	    goto send_again;
	}
      else
	{
	  /* xatgs == NULL.  Collect a response without sending a
	     query.  In this mode, we need to ignore the total
	     deadline.  */
	  milliseconds = __deadline_to_ms (current_time, response_deadline);
	  if (milliseconds == 0)
	    /* Cannot send again, so bail out.  */
	    return (cu->cu_error.re_status = RPC_CANTSEND);
	}

      switch (__poll (&fd, 1, milliseconds))
	{

	case 0:
	  if (anyup == 0)
	    {
	      anyup = is_network_up (cu->cu_sock);
	      if (!anyup)
		return (cu->cu_error.re_status = RPC_CANTRECV);
	    }
	  goto next_response;
	case -1:
	  if (errno == EINTR)
	    goto next_response;
	  cu->cu_error.re_errno = errno;
	  return (cu->cu_error.re_status = RPC_CANTRECV);
	}
#ifdef IP_RECVERR
      if (fd.revents & POLLERR)
	{
	  struct msghdr msg;
	  struct cmsghdr *cmsg;
	  struct sock_extended_err *e;
	  struct sockaddr_in err_addr;
	  struct iovec iov;
	  char *cbuf = malloc (outlen + 256);
	  int ret;

	  if (cbuf == NULL)
	    {
	      cu->cu_error.re_errno = errno;
	      return (cu->cu_error.re_status = RPC_CANTRECV);
	    }

	  iov.iov_base = cbuf + 256;
	  iov.iov_len = outlen;
	  msg.msg_name = (void *) &err_addr;
	  msg.msg_namelen = sizeof (err_addr);
	  msg.msg_iov = &iov;
	  msg.msg_iovlen = 1;
	  msg.msg_flags = 0;
	  msg.msg_control = cbuf;
	  msg.msg_controllen = 256;
	  ret = __recvmsg (cu->cu_sock, &msg, MSG_ERRQUEUE);
	  if (ret >= 0
	      && memcmp (cbuf + 256, cu->cu_outbuf, ret) == 0
	      && (msg.msg_flags & MSG_ERRQUEUE)
	      && ((msg.msg_namelen == 0
		   && ret >= 12)
		  || (msg.msg_namelen == sizeof (err_addr)
		      && err_addr.sin_family == AF_INET
		      && memcmp (&err_addr.sin_addr, &cu->cu_raddr.sin_addr,
				 sizeof (err_addr.sin_addr)) == 0
		      && err_addr.sin_port == cu->cu_raddr.sin_port)))
	    for (cmsg = CMSG_FIRSTHDR (&msg); cmsg;
		 cmsg = CMSG_NXTHDR (&msg, cmsg))
	      if (cmsg->cmsg_level == SOL_IP && cmsg->cmsg_type == IP_RECVERR)
		{
		  e = (struct sock_extended_err *) CMSG_DATA(cmsg);
		  cu->cu_error.re_errno = e->ee_errno;
		  free (cbuf);
		  return (cu->cu_error.re_status = RPC_CANTRECV);
		}
	  free (cbuf);
	}
#endif
      do
	{
	  fromlen = sizeof (struct sockaddr);
	  inlen = __recvfrom (cu->cu_sock, cu->cu_inbuf,
			      (int) cu->cu_recvsz, MSG_DONTWAIT,
			      (struct sockaddr *) &from, &fromlen);
	}
      while (inlen < 0 && errno == EINTR);
      if (inlen < 0)
	{
	  if (errno == EWOULDBLOCK)
	    goto next_response;
	  cu->cu_error.re_errno = errno;
	  return (cu->cu_error.re_status = RPC_CANTRECV);
	}
      /* Accept the response if the packet is sufficiently long and
	 the transaction ID matches the query (if available).  */
      if (inlen >= 4
	  && (xargs == NULL
	      || memcmp (cu->cu_inbuf, cu->cu_outbuf,
			 sizeof (uint32_t)) == 0))
	break;

    next_response:
      /* Update the current time because poll and recvmsg waited for
	 an unknown time.  */
      current_time = __deadline_current_time ();
    }

  /*
   * now decode and validate the response
   */
  xdrmem_create (&reply_xdrs, cu->cu_inbuf, (u_int) inlen, XDR_DECODE);
  ok = xdr_replymsg (&reply_xdrs, &reply_msg);
  /* XDR_DESTROY(&reply_xdrs);  save a few cycles on noop destroy */
  if (ok)
    {
      _seterr_reply (&reply_msg, &(cu->cu_error));
      if (cu->cu_error.re_status == RPC_SUCCESS)
	{
	  if (!AUTH_VALIDATE (cl->cl_auth,
			      &reply_msg.acpted_rply.ar_verf))
	    {
	      cu->cu_error.re_status = RPC_AUTHERROR;
	      cu->cu_error.re_why = AUTH_INVALIDRESP;
	    }
	  if (reply_msg.acpted_rply.ar_verf.oa_base != NULL)
	    {
	      xdrs->x_op = XDR_FREE;
	      (void) xdr_opaque_auth (xdrs, &(reply_msg.acpted_rply.ar_verf));
	    }
	}			/* end successful completion */
      else
	{
	  /* maybe our credentials need to be refreshed ... */
	  if (nrefreshes > 0 && AUTH_REFRESH (cl->cl_auth))
	    {
	      nrefreshes--;
	      goto call_again;
	    }
	}			/* end of unsuccessful completion */
    }				/* end of valid reply message */
  else
    {
      cu->cu_error.re_status = RPC_CANTDECODERES;
    }
  return cu->cu_error.re_status;
}

static void
clntudp_geterr (CLIENT *cl, struct rpc_err *errp)
{
  struct cu_data *cu = (struct cu_data *) cl->cl_private;

  *errp = cu->cu_error;
}


static bool_t
clntudp_freeres (CLIENT *cl, xdrproc_t xdr_res, caddr_t res_ptr)
{
  struct cu_data *cu = (struct cu_data *) cl->cl_private;
  XDR *xdrs = &(cu->cu_outxdrs);

  xdrs->x_op = XDR_FREE;
  return (*xdr_res) (xdrs, res_ptr);
}

static void
clntudp_abort (void)
{
}

static bool_t
clntudp_control (CLIENT *cl, int request, char *info)
{
  struct cu_data *cu = (struct cu_data *) cl->cl_private;
  u_long ul;
  uint32_t ui32;

  switch (request)
    {
    case CLSET_FD_CLOSE:
      cu->cu_closeit = TRUE;
      break;
    case CLSET_FD_NCLOSE:
      cu->cu_closeit = FALSE;
      break;
    case CLSET_TIMEOUT:
      cu->cu_total = *(struct timeval *) info;
      break;
    case CLGET_TIMEOUT:
      *(struct timeval *) info = cu->cu_total;
      break;
    case CLSET_RETRY_TIMEOUT:
      cu->cu_wait = *(struct timeval *) info;
      break;
    case CLGET_RETRY_TIMEOUT:
      *(struct timeval *) info = cu->cu_wait;
      break;
    case CLGET_SERVER_ADDR:
      *(struct sockaddr_in *) info = cu->cu_raddr;
      break;
    case CLGET_FD:
      *(int *)info = cu->cu_sock;
      break;
    case CLGET_XID:
      /*
       * use the knowledge that xid is the
       * first element in the call structure *.
       * This will get the xid of the PREVIOUS call
       */
      memcpy (&ui32, cu->cu_outbuf, sizeof (ui32));
      ul = ntohl (ui32);
      memcpy (info, &ul, sizeof (ul));
      break;
    case CLSET_XID:
      /* This will set the xid of the NEXT call */
      memcpy (&ul, info, sizeof (ul));
      ui32 = htonl (ul - 1);
      memcpy (cu->cu_outbuf, &ui32, sizeof (ui32));
      /* decrement by 1 as clntudp_call() increments once */
      break;
    case CLGET_VERS:
      /*
       * This RELIES on the information that, in the call body,
       * the version number field is the fifth field from the
       * beginning of the RPC header. MUST be changed if the
       * call_struct is changed
       */
      memcpy (&ui32, cu->cu_outbuf + 4 * BYTES_PER_XDR_UNIT, sizeof (ui32));
      ul = ntohl (ui32);
      memcpy (info, &ul, sizeof (ul));
      break;
    case CLSET_VERS:
      memcpy (&ul, info, sizeof (ul));
      ui32 = htonl (ul);
      memcpy (cu->cu_outbuf + 4 * BYTES_PER_XDR_UNIT, &ui32, sizeof (ui32));
      break;
    case CLGET_PROG:
      /*
       * This RELIES on the information that, in the call body,
       * the program number field is the  field from the
       * beginning of the RPC header. MUST be changed if the
       * call_struct is changed
       */
      memcpy (&ui32, cu->cu_outbuf + 3 * BYTES_PER_XDR_UNIT, sizeof (ui32));
      ul = ntohl (ui32);
      memcpy (info, &ul, sizeof (ul));
      break;
    case CLSET_PROG:
      memcpy (&ul, info, sizeof (ul));
      ui32 = htonl (ul);
      memcpy (cu->cu_outbuf + 3 * BYTES_PER_XDR_UNIT, &ui32, sizeof (ui32));
      break;
    /* The following are only possible with TI-RPC */
    case CLGET_SVC_ADDR:
    case CLSET_SVC_ADDR:
    case CLSET_PUSH_TIMOD:
    case CLSET_POP_TIMOD:
    default:
      return FALSE;
    }
  return TRUE;
}

static void
clntudp_destroy (CLIENT *cl)
{
  struct cu_data *cu = (struct cu_data *) cl->cl_private;

  if (cu->cu_closeit)
    {
      (void) __close (cu->cu_sock);
    }
  XDR_DESTROY (&(cu->cu_outxdrs));
  mem_free ((caddr_t) cu, (sizeof (*cu) + cu->cu_sendsz + cu->cu_recvsz));
  mem_free ((caddr_t) cl, sizeof (CLIENT));
}
