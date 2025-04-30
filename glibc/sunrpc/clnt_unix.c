/*
 * clnt_unix.c, Implements a TCP/IP based, client side RPC.
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
 *
 * TCP based RPC supports 'batched calls'.
 * A sequence of calls may be batched-up in a send buffer.  The rpc call
 * return immediately to the client even though the call was not necessarily
 * sent.  The batching occurs if the results' xdr routine is NULL (0) AND
 * the rpc timeout value is zero (see clnt.h, rpc).
 *
 * Clients should NOT casually batch calls that in fact return results; that is,
 * the server side should be aware that a call is batched and not produce any
 * return message.  Batched calls that produce many result messages can
 * deadlock (netlock) the client and the server....
 *
 * Now go hang yourself.
 */

#include <netdb.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <libintl.h>
#include <rpc/rpc.h>
#include <sys/uio.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <rpc/pmap_clnt.h>
#include <wchar.h>
#include <shlib-compat.h>

extern u_long _create_xid (void);

#define MCALL_MSG_SIZE 24

struct ct_data
  {
    int ct_sock;
    bool_t ct_closeit;
    struct timeval ct_wait;
    bool_t ct_waitset;		/* wait set by clnt_control? */
    struct sockaddr_un ct_addr;
    struct rpc_err ct_error;
    char ct_mcall[MCALL_MSG_SIZE];	/* marshalled callmsg */
    u_int ct_mpos;		/* pos after marshal */
    XDR ct_xdrs;
  };

static int readunix (char *, char *, int);
static int writeunix (char *, char *, int);

static enum clnt_stat clntunix_call (CLIENT *, u_long, xdrproc_t, caddr_t,
				    xdrproc_t, caddr_t, struct timeval);
static void clntunix_abort (void);
static void clntunix_geterr (CLIENT *, struct rpc_err *);
static bool_t clntunix_freeres (CLIENT *, xdrproc_t, caddr_t);
static bool_t clntunix_control (CLIENT *, int, char *);
static void clntunix_destroy (CLIENT *);

static const struct clnt_ops unix_ops =
{
  clntunix_call,
  clntunix_abort,
  clntunix_geterr,
  clntunix_freeres,
  clntunix_destroy,
  clntunix_control
};

/*
 * Create a client handle for a tcp/ip connection.
 * If *sockp<0, *sockp is set to a newly created TCP socket and it is
 * connected to raddr.  If *sockp non-negative then
 * raddr is ignored.  The rpc/tcp package does buffering
 * similar to stdio, so the client must pick send and receive buffer sizes,];
 * 0 => use the default.
 * If raddr->sin_port is 0, then a binder on the remote machine is
 * consulted for the right port number.
 * NB: *sockp is copied into a private area.
 * NB: It is the clients responsibility to close *sockp.
 * NB: The rpch->cl_auth is set null authentication.  Caller may wish to set this
 * something more useful.
 */
CLIENT *
clntunix_create (struct sockaddr_un *raddr, u_long prog, u_long vers,
		 int *sockp, u_int sendsz, u_int recvsz)
{
  CLIENT *h;
  struct ct_data *ct = (struct ct_data *) mem_alloc (sizeof (*ct));
  struct rpc_msg call_msg;
  int len;

  h = (CLIENT *) mem_alloc (sizeof (*h));
  if (h == NULL || ct == NULL)
    {
      struct rpc_createerr *ce = &get_rpc_createerr ();
      (void) __fxprintf (NULL, "%s: %s", __func__, _("out of memory\n"));
      ce->cf_stat = RPC_SYSTEMERROR;
      ce->cf_error.re_errno = ENOMEM;
      goto fooy;
    }

  /*
   * If no socket given, open one
   */
  if (*sockp < 0)
    {
      *sockp = __socket (AF_UNIX, SOCK_STREAM, 0);
      len = strlen (raddr->sun_path) + sizeof (raddr->sun_family) + 1;
      if (*sockp < 0
	  || __connect (*sockp, (struct sockaddr *) raddr, len) < 0)
	{
	  struct rpc_createerr *ce = &get_rpc_createerr ();
	  ce->cf_stat = RPC_SYSTEMERROR;
	  ce->cf_error.re_errno = errno;
	  if (*sockp != -1)
	    __close (*sockp);
	  goto fooy;
	}
      ct->ct_closeit = TRUE;
    }
  else
    {
      ct->ct_closeit = FALSE;
    }

  /*
   * Set up private data struct
   */
  ct->ct_sock = *sockp;
  ct->ct_wait.tv_usec = 0;
  ct->ct_waitset = FALSE;
  ct->ct_addr = *raddr;

  /*
   * Initialize call message
   */
  call_msg.rm_xid = _create_xid ();
  call_msg.rm_direction = CALL;
  call_msg.rm_call.cb_rpcvers = RPC_MSG_VERSION;
  call_msg.rm_call.cb_prog = prog;
  call_msg.rm_call.cb_vers = vers;

  /*
   * pre-serialize the static part of the call msg and stash it away
   */
  xdrmem_create (&(ct->ct_xdrs), ct->ct_mcall, MCALL_MSG_SIZE, XDR_ENCODE);
  if (!xdr_callhdr (&(ct->ct_xdrs), &call_msg))
    {
      if (ct->ct_closeit)
	__close (*sockp);
      goto fooy;
    }
  ct->ct_mpos = XDR_GETPOS (&(ct->ct_xdrs));
  XDR_DESTROY (&(ct->ct_xdrs));

  /*
   * Create a client handle which uses xdrrec for serialization
   * and authnone for authentication.
   */
  xdrrec_create (&(ct->ct_xdrs), sendsz, recvsz,
		 (caddr_t) ct, readunix, writeunix);
  h->cl_ops = (struct clnt_ops *) &unix_ops;
  h->cl_private = (caddr_t) ct;
  h->cl_auth = authnone_create ();
  return h;

fooy:
  /*
   * Something goofed, free stuff and barf
   */
  mem_free ((caddr_t) ct, sizeof (struct ct_data));
  mem_free ((caddr_t) h, sizeof (CLIENT));
  return (CLIENT *) NULL;
}
libc_hidden_nolink_sunrpc (clntunix_create, GLIBC_2_1)

static enum clnt_stat
clntunix_call (CLIENT *h, u_long proc, xdrproc_t xdr_args, caddr_t args_ptr,
	       xdrproc_t xdr_results, caddr_t results_ptr,
	       struct timeval timeout)
{
  struct ct_data *ct = (struct ct_data *) h->cl_private;
  XDR *xdrs = &(ct->ct_xdrs);
  struct rpc_msg reply_msg;
  u_long x_id;
  uint32_t *msg_x_id = (uint32_t *) (ct->ct_mcall);	/* yuk */
  bool_t shipnow;
  int refreshes = 2;

  if (!ct->ct_waitset)
    {
      ct->ct_wait = timeout;
    }

  shipnow =
    (xdr_results == (xdrproc_t) 0 && ct->ct_wait.tv_sec == 0
     && ct->ct_wait.tv_usec == 0) ? FALSE : TRUE;

call_again:
  xdrs->x_op = XDR_ENCODE;
  ct->ct_error.re_status = RPC_SUCCESS;
  x_id = ntohl (--(*msg_x_id));
  if ((!XDR_PUTBYTES (xdrs, ct->ct_mcall, ct->ct_mpos)) ||
      (!XDR_PUTLONG (xdrs, (long *) &proc)) ||
      (!AUTH_MARSHALL (h->cl_auth, xdrs)) ||
      (!(*xdr_args) (xdrs, args_ptr)))
    {
      if (ct->ct_error.re_status == RPC_SUCCESS)
	ct->ct_error.re_status = RPC_CANTENCODEARGS;
      (void) xdrrec_endofrecord (xdrs, TRUE);
      return ct->ct_error.re_status;
    }
  if (!xdrrec_endofrecord (xdrs, shipnow))
    return ct->ct_error.re_status = RPC_CANTSEND;
  if (!shipnow)
    return RPC_SUCCESS;
  /*
   * Hack to provide rpc-based message passing
   */
  if (ct->ct_wait.tv_sec == 0 && ct->ct_wait.tv_usec == 0)
    return ct->ct_error.re_status = RPC_TIMEDOUT;


  /*
   * Keep receiving until we get a valid transaction id
   */
  xdrs->x_op = XDR_DECODE;
  while (TRUE)
    {
      reply_msg.acpted_rply.ar_verf = _null_auth;
      reply_msg.acpted_rply.ar_results.where = NULL;
      reply_msg.acpted_rply.ar_results.proc = (xdrproc_t)xdr_void;
      if (!xdrrec_skiprecord (xdrs))
	return ct->ct_error.re_status;
      /* now decode and validate the response header */
      if (!xdr_replymsg (xdrs, &reply_msg))
	{
	  if (ct->ct_error.re_status == RPC_SUCCESS)
	    continue;
	  return ct->ct_error.re_status;
	}
      if (reply_msg.rm_xid == x_id)
	break;
    }

  /*
   * process header
   */
  _seterr_reply (&reply_msg, &(ct->ct_error));
  if (ct->ct_error.re_status == RPC_SUCCESS)
    {
      if (!AUTH_VALIDATE (h->cl_auth, &reply_msg.acpted_rply.ar_verf))
	{
	  ct->ct_error.re_status = RPC_AUTHERROR;
	  ct->ct_error.re_why = AUTH_INVALIDRESP;
	}
      else if (!(*xdr_results) (xdrs, results_ptr))
	{
	  if (ct->ct_error.re_status == RPC_SUCCESS)
	    ct->ct_error.re_status = RPC_CANTDECODERES;
	}
      /* free verifier ... */
      if (reply_msg.acpted_rply.ar_verf.oa_base != NULL)
	{
	  xdrs->x_op = XDR_FREE;
	  (void) xdr_opaque_auth (xdrs, &(reply_msg.acpted_rply.ar_verf));
	}
    }				/* end successful completion */
  else
    {
      /* maybe our credentials need to be refreshed ... */
      if (refreshes-- && AUTH_REFRESH (h->cl_auth))
	goto call_again;
    }				/* end of unsuccessful completion */
  return ct->ct_error.re_status;
}

static void
clntunix_geterr (CLIENT *h, struct rpc_err *errp)
{
  struct ct_data *ct = (struct ct_data *) h->cl_private;

  *errp = ct->ct_error;
}

static bool_t
clntunix_freeres (CLIENT *cl, xdrproc_t xdr_res, caddr_t res_ptr)
{
  struct ct_data *ct = (struct ct_data *) cl->cl_private;
  XDR *xdrs = &(ct->ct_xdrs);

  xdrs->x_op = XDR_FREE;
  return (*xdr_res) (xdrs, res_ptr);
}

static void
clntunix_abort (void)
{
}

static bool_t
clntunix_control (CLIENT *cl, int request, char *info)
{
  struct ct_data *ct = (struct ct_data *) cl->cl_private;
  u_long ul;
  uint32_t ui32;

  switch (request)
    {
    case CLSET_FD_CLOSE:
      ct->ct_closeit = TRUE;
      break;
    case CLSET_FD_NCLOSE:
      ct->ct_closeit = FALSE;
      break;
    case CLSET_TIMEOUT:
      ct->ct_wait = *(struct timeval *) info;
      break;
    case CLGET_TIMEOUT:
      *(struct timeval *) info = ct->ct_wait;
      break;
    case CLGET_SERVER_ADDR:
      *(struct sockaddr_un *) info = ct->ct_addr;
      break;
    case CLGET_FD:
      *(int *)info = ct->ct_sock;
      break;
    case CLGET_XID:
      /*
       * use the knowledge that xid is the
       * first element in the call structure *.
       * This will get the xid of the PREVIOUS call
       */
      memcpy (&ui32, ct->ct_mcall, sizeof (ui32));
      ul = ntohl (ui32);
      memcpy (info, &ul, sizeof (ul));
      break;
    case CLSET_XID:
      /* This will set the xid of the NEXT call */
      memcpy (&ul, info, sizeof (ul));
      ui32 = htonl (ul - 1);
      memcpy (ct->ct_mcall, &ui32, sizeof (ui32));
      /* decrement by 1 as clntunix_call() increments once */
      break;
    case CLGET_VERS:
      /*
       * This RELIES on the information that, in the call body,
       * the version number field is the fifth field from the
       * beginning of the RPC header. MUST be changed if the
       * call_struct is changed
       */
      memcpy (&ui32, ct->ct_mcall + 4 * BYTES_PER_XDR_UNIT, sizeof (ui32));
      ul = ntohl (ui32);
      memcpy (info, &ul, sizeof (ul));
      break;
    case CLSET_VERS:
      memcpy (&ul, info, sizeof (ul));
      ui32 = htonl (ul);
      memcpy (ct->ct_mcall + 4 * BYTES_PER_XDR_UNIT, &ui32, sizeof (ui32));
      break;
    case CLGET_PROG:
      /*
       * This RELIES on the information that, in the call body,
       * the program number field is the  field from the
       * beginning of the RPC header. MUST be changed if the
       * call_struct is changed
       */
      memcpy (&ui32, ct->ct_mcall + 3 * BYTES_PER_XDR_UNIT, sizeof (ui32));
      ul = ntohl (ui32);
      memcpy (info, &ul, sizeof (ul));
      break;
    case CLSET_PROG:
      memcpy (&ul, info, sizeof (ul));
      ui32 = htonl (ul);
      memcpy (ct->ct_mcall + 3 * BYTES_PER_XDR_UNIT, &ui32, sizeof (ui32));
      break;
    /* The following are only possible with TI-RPC */
    case CLGET_RETRY_TIMEOUT:
    case CLSET_RETRY_TIMEOUT:
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
clntunix_destroy (CLIENT *h)
{
  struct ct_data *ct =
  (struct ct_data *) h->cl_private;

  if (ct->ct_closeit)
    {
      (void) __close (ct->ct_sock);
    }
  XDR_DESTROY (&(ct->ct_xdrs));
  mem_free ((caddr_t) ct, sizeof (struct ct_data));
  mem_free ((caddr_t) h, sizeof (CLIENT));
}

static int
__msgread (int sock, void *data, size_t cnt)
{
  struct iovec iov;
  struct msghdr msg;
#ifdef SCM_CREDENTIALS
  static char cm[CMSG_SPACE(sizeof (struct ucred))];
#endif
  int len;

  iov.iov_base = data;
  iov.iov_len = cnt;

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_name = NULL;
  msg.msg_namelen = 0;
#ifdef SCM_CREDENTIALS
  msg.msg_control = (caddr_t) &cm;
  msg.msg_controllen = CMSG_SPACE(sizeof (struct ucred));
#endif
  msg.msg_flags = 0;

#ifdef SO_PASSCRED
  {
    int on = 1;
    if (__setsockopt (sock, SOL_SOCKET, SO_PASSCRED, &on, sizeof (on)))
      return -1;
  }
#endif

 restart:
  len = __recvmsg (sock, &msg, 0);
  if (len >= 0)
    {
      if (msg.msg_flags & MSG_CTRUNC || len == 0)
	return 0;
      else
	return len;
    }
  if (errno == EINTR)
    goto restart;
  return -1;
}

static int
__msgwrite (int sock, void *data, size_t cnt)
{
#ifndef SCM_CREDENTIALS
  /* We cannot implement this reliably.  */
  __set_errno (ENOSYS);
  return -1;
#else
  struct iovec iov;
  struct msghdr msg;
  struct cmsghdr *cmsg = alloca (CMSG_SPACE(sizeof (struct ucred)));
  struct ucred cred;
  int len;

  /* XXX I'm not sure, if gete?id() is always correct, or if we should use
     get?id(). But since keyserv needs geteuid(), we have no other chance.
     It would be much better, if the kernel could pass both to the server. */
  cred.pid = __getpid ();
  cred.uid = __geteuid ();
  cred.gid = __getegid ();

  memcpy (CMSG_DATA(cmsg), &cred, sizeof (struct ucred));
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_CREDENTIALS;
  cmsg->cmsg_len = sizeof(*cmsg) + sizeof(struct ucred);

  iov.iov_base = data;
  iov.iov_len = cnt;

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_name = NULL;
  msg.msg_namelen = 0;
  msg.msg_control = cmsg;
  msg.msg_controllen = CMSG_ALIGN(cmsg->cmsg_len);
  msg.msg_flags = 0;

 restart:
  len = __sendmsg (sock, &msg, 0);
  if (len >= 0)
    return len;
  if (errno == EINTR)
    goto restart;
  return -1;

#endif
}


/*
 * Interface between xdr serializer and unix connection.
 * Behaves like the system calls, read & write, but keeps some error state
 * around for the rpc level.
 */
static int
readunix (char *ctptr, char *buf, int len)
{
  struct ct_data *ct = (struct ct_data *) ctptr;
  struct pollfd fd;
  int milliseconds = ((ct->ct_wait.tv_sec * 1000)
		      + (ct->ct_wait.tv_usec / 1000));

  if (len == 0)
    return 0;

  fd.fd = ct->ct_sock;
  fd.events = POLLIN;
  while (TRUE)
    {
      switch (__poll (&fd, 1, milliseconds))
	{
	case 0:
	  ct->ct_error.re_status = RPC_TIMEDOUT;
	  return -1;

	case -1:
	  if (errno == EINTR)
	    continue;
	  ct->ct_error.re_status = RPC_CANTRECV;
	  ct->ct_error.re_errno = errno;
	  return -1;
	}
      break;
    }
  switch (len = __msgread (ct->ct_sock, buf, len))
    {

    case 0:
      /* premature eof */
      ct->ct_error.re_errno = ECONNRESET;
      ct->ct_error.re_status = RPC_CANTRECV;
      len = -1;			/* it's really an error */
      break;

    case -1:
      ct->ct_error.re_errno = errno;
      ct->ct_error.re_status = RPC_CANTRECV;
      break;
    }
  return len;
}

static int
writeunix (char *ctptr, char *buf, int len)
{
  int i, cnt;
  struct ct_data *ct = (struct ct_data *) ctptr;

  for (cnt = len; cnt > 0; cnt -= i, buf += i)
    {
      if ((i = __msgwrite (ct->ct_sock, buf, cnt)) == -1)
	{
	  ct->ct_error.re_errno = errno;
	  ct->ct_error.re_status = RPC_CANTSEND;
	  return -1;
	}
    }
  return len;
}
