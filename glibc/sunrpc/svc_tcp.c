/*
 * svc_tcp.c, Server side for TCP/IP based RPC.
 *
 * Copyright (C) 2012-2021 Free Software Foundation, Inc.
 * This file is part of the GNU C Library.
 *
 * The GNU C Library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * The GNU C Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with the GNU C Library; if not, see
 * <https://www.gnu.org/licenses/>.
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
 * Actually implements two flavors of transporter -
 * a tcp rendezvouser (a listener and connection establisher)
 * and a record/tcp stream.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <libintl.h>
#include <rpc/rpc.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <errno.h>
#include <stdlib.h>

#include <wchar.h>
#include <libio/iolibio.h>
#include <shlib-compat.h>

/*
 * Ops vector for TCP/IP based rpc service handle
 */
static bool_t svctcp_recv (SVCXPRT *, struct rpc_msg *);
static enum xprt_stat svctcp_stat (SVCXPRT *);
static bool_t svctcp_getargs (SVCXPRT *, xdrproc_t, caddr_t);
static bool_t svctcp_reply (SVCXPRT *, struct rpc_msg *);
static bool_t svctcp_freeargs (SVCXPRT *, xdrproc_t, caddr_t);
static void svctcp_destroy (SVCXPRT *);

static const struct xp_ops svctcp_op =
{
  svctcp_recv,
  svctcp_stat,
  svctcp_getargs,
  svctcp_reply,
  svctcp_freeargs,
  svctcp_destroy
};

/*
 * Ops vector for TCP/IP rendezvous handler
 */
static bool_t rendezvous_request (SVCXPRT *, struct rpc_msg *);
static enum xprt_stat rendezvous_stat (SVCXPRT *);
static void svctcp_rendezvous_abort (void) __attribute__ ((__noreturn__));

/* This function makes sure abort() relocation goes through PLT
   and thus can be lazy bound.  */
static void
svctcp_rendezvous_abort (void)
{
  abort ();
};

static const struct xp_ops svctcp_rendezvous_op =
{
  rendezvous_request,
  rendezvous_stat,
  (bool_t (*) (SVCXPRT *, xdrproc_t, caddr_t)) svctcp_rendezvous_abort,
  (bool_t (*) (SVCXPRT *, struct rpc_msg *)) svctcp_rendezvous_abort,
  (bool_t (*) (SVCXPRT *, xdrproc_t, caddr_t)) svctcp_rendezvous_abort,
  svctcp_destroy
};

static int readtcp (char*, char *, int);
static int writetcp (char *, char *, int);
static SVCXPRT *makefd_xprt (int, u_int, u_int);

struct tcp_rendezvous
  {				/* kept in xprt->xp_p1 */
    u_int sendsize;
    u_int recvsize;
  };

struct tcp_conn
  {				/* kept in xprt->xp_p1 */
    enum xprt_stat strm_stat;
    u_long x_id;
    XDR xdrs;
    char verf_body[MAX_AUTH_BYTES];
  };

/*
 * Usage:
 *      xprt = svctcp_create(sock, send_buf_size, recv_buf_size);
 *
 * Creates, registers, and returns a (rpc) tcp based transporter.
 * Once *xprt is initialized, it is registered as a transporter
 * see (svc.h, xprt_register).  This routine returns
 * a NULL if a problem occurred.
 *
 * If sock<0 then a socket is created, else sock is used.
 * If the socket, sock is not bound to a port then svctcp_create
 * binds it to an arbitrary port.  The routine then starts a tcp
 * listener on the socket's associated port.  In any (successful) case,
 * xprt->xp_sock is the registered socket number and xprt->xp_port is the
 * associated port number.
 *
 * Since tcp streams do buffered io similar to stdio, the caller can specify
 * how big the send and receive buffers are via the second and third parms;
 * 0 => use the system default.
 */
SVCXPRT *
svctcp_create (int sock, u_int sendsize, u_int recvsize)
{
  bool_t madesock = FALSE;
  SVCXPRT *xprt;
  struct tcp_rendezvous *r;
  struct sockaddr_in addr;
  socklen_t len = sizeof (struct sockaddr_in);

  if (sock == RPC_ANYSOCK)
    {
      if ((sock = __socket (AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
	{
	  perror (_("svc_tcp.c - tcp socket creation problem"));
	  return (SVCXPRT *) NULL;
	}
      madesock = TRUE;
    }
  memset ((char *) &addr, 0, sizeof (addr));
  addr.sin_family = AF_INET;
  if (bindresvport (sock, &addr))
    {
      addr.sin_port = 0;
      (void) __bind (sock, (struct sockaddr *) &addr, len);
    }
  if ((__getsockname (sock, (struct sockaddr *) &addr, &len) != 0) ||
      (__listen (sock, SOMAXCONN) != 0))
    {
      perror (_("svc_tcp.c - cannot getsockname or listen"));
      if (madesock)
	(void) __close (sock);
      return (SVCXPRT *) NULL;
    }
  r = (struct tcp_rendezvous *) mem_alloc (sizeof (*r));
  xprt = (SVCXPRT *) mem_alloc (sizeof (SVCXPRT));
  if (r == NULL || xprt == NULL)
    {
      (void) __fxprintf (NULL, "%s: %s", __func__, _("out of memory\n"));
      mem_free (r, sizeof (*r));
      mem_free (xprt, sizeof (SVCXPRT));
      return NULL;
    }
  r->sendsize = sendsize;
  r->recvsize = recvsize;
  xprt->xp_p2 = NULL;
  xprt->xp_p1 = (caddr_t) r;
  xprt->xp_verf = _null_auth;
  xprt->xp_ops = &svctcp_rendezvous_op;
  xprt->xp_port = ntohs (addr.sin_port);
  xprt->xp_sock = sock;
  xprt_register (xprt);
  return xprt;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svctcp_create)
#else
libc_hidden_nolink_sunrpc (svctcp_create, GLIBC_2_0)
#endif

/*
 * Like svtcp_create(), except the routine takes any *open* UNIX file
 * descriptor as its first input.
 */
SVCXPRT *
svcfd_create (int fd, u_int sendsize, u_int recvsize)
{
  return makefd_xprt (fd, sendsize, recvsize);
}
libc_hidden_nolink_sunrpc (svcfd_create, GLIBC_2_0)

static SVCXPRT *
makefd_xprt (int fd, u_int sendsize, u_int recvsize)
{
  SVCXPRT *xprt;
  struct tcp_conn *cd;

  xprt = (SVCXPRT *) mem_alloc (sizeof (SVCXPRT));
  cd = (struct tcp_conn *) mem_alloc (sizeof (struct tcp_conn));
  if (xprt == (SVCXPRT *) NULL || cd == NULL)
    {
      (void) __fxprintf (NULL, "%s: %s", "svc_tcp: makefd_xprt",
			 _("out of memory\n"));
      mem_free (xprt, sizeof (SVCXPRT));
      mem_free (cd, sizeof (struct tcp_conn));
      return NULL;
    }
  cd->strm_stat = XPRT_IDLE;
  xdrrec_create (&(cd->xdrs), sendsize, recvsize,
		 (caddr_t) xprt, readtcp, writetcp);
  xprt->xp_p2 = NULL;
  xprt->xp_p1 = (caddr_t) cd;
  xprt->xp_verf.oa_base = cd->verf_body;
  xprt->xp_addrlen = 0;
  xprt->xp_ops = &svctcp_op;	/* truly deals with calls */
  xprt->xp_port = 0;		/* this is a connection, not a rendezvouser */
  xprt->xp_sock = fd;
  xprt_register (xprt);
  return xprt;
}

static bool_t
rendezvous_request (SVCXPRT *xprt, struct rpc_msg *errmsg)
{
  int sock;
  struct tcp_rendezvous *r;
  struct sockaddr_in addr;
  socklen_t len;

  r = (struct tcp_rendezvous *) xprt->xp_p1;
again:
  len = sizeof (struct sockaddr_in);
  if ((sock = accept (xprt->xp_sock, (struct sockaddr *) &addr, &len)) < 0)
    {
      if (errno == EINTR)
	goto again;
      __svc_accept_failed ();
      return FALSE;
    }
  /*
   * make a new transporter (re-uses xprt)
   */
  xprt = makefd_xprt (sock, r->sendsize, r->recvsize);

  /* If we are out of memory, makefd_xprt has already dumped an error.  */
  if (xprt == NULL)
    {
      __svc_wait_on_error ();
      return FALSE;
    }

  memcpy (&xprt->xp_raddr, &addr, sizeof (addr));
  xprt->xp_addrlen = len;
  return FALSE;		/* there is never an rpc msg to be processed */
}

static enum xprt_stat
rendezvous_stat (SVCXPRT *xprt)
{
  return XPRT_IDLE;
}

static void
svctcp_destroy (SVCXPRT *xprt)
{
  struct tcp_conn *cd = (struct tcp_conn *) xprt->xp_p1;

  xprt_unregister (xprt);
  (void) __close (xprt->xp_sock);
  if (xprt->xp_port != 0)
    {
      /* a rendezvouser socket */
      xprt->xp_port = 0;
    }
  else
    {
      /* an actual connection socket */
      XDR_DESTROY (&(cd->xdrs));
    }
  mem_free ((caddr_t) cd, sizeof (struct tcp_conn));
  mem_free ((caddr_t) xprt, sizeof (SVCXPRT));
}


/*
 * reads data from the tcp connection.
 * any error is fatal and the connection is closed.
 * (And a read of zero bytes is a half closed stream => error.)
 */
static int
readtcp (char *xprtptr, char *buf, int len)
{
  SVCXPRT *xprt = (SVCXPRT *)xprtptr;
  int sock = xprt->xp_sock;
  int milliseconds = 35 * 1000;
  struct pollfd pollfd;

  do
    {
      pollfd.fd = sock;
      pollfd.events = POLLIN;
      switch (__poll (&pollfd, 1, milliseconds))
	{
	case -1:
	  if (errno == EINTR)
	    continue;
	  /*FALLTHROUGH*/
	case 0:
	  goto fatal_err;
	default:
	  if ((pollfd.revents & POLLERR) || (pollfd.revents & POLLHUP)
	      || (pollfd.revents & POLLNVAL))
	    goto fatal_err;
	  break;
	}
    }
  while ((pollfd.revents & POLLIN) == 0);

  if ((len = __read (sock, buf, len)) > 0)
    return len;

 fatal_err:
  ((struct tcp_conn *) (xprt->xp_p1))->strm_stat = XPRT_DIED;
  return -1;
}

/*
 * writes data to the tcp connection.
 * Any error is fatal and the connection is closed.
 */
static int
writetcp (char *xprtptr, char * buf, int len)
{
  SVCXPRT *xprt = (SVCXPRT *)xprtptr;
  int i, cnt;

  for (cnt = len; cnt > 0; cnt -= i, buf += i)
    {
      if ((i = __write (xprt->xp_sock, buf, cnt)) < 0)
	{
	  ((struct tcp_conn *) (xprt->xp_p1))->strm_stat = XPRT_DIED;
	  return -1;
	}
    }
  return len;
}

static enum xprt_stat
svctcp_stat (SVCXPRT *xprt)
{
  struct tcp_conn *cd =
  (struct tcp_conn *) (xprt->xp_p1);

  if (cd->strm_stat == XPRT_DIED)
    return XPRT_DIED;
  if (!xdrrec_eof (&(cd->xdrs)))
    return XPRT_MOREREQS;
  return XPRT_IDLE;
}

static bool_t
svctcp_recv (SVCXPRT *xprt, struct rpc_msg *msg)
{
  struct tcp_conn *cd = (struct tcp_conn *) (xprt->xp_p1);
  XDR *xdrs = &(cd->xdrs);

  xdrs->x_op = XDR_DECODE;
  (void) xdrrec_skiprecord (xdrs);
  if (xdr_callmsg (xdrs, msg))
    {
      cd->x_id = msg->rm_xid;
      return TRUE;
    }
  cd->strm_stat = XPRT_DIED;	/* XXXX */
  return FALSE;
}

static bool_t
svctcp_getargs (SVCXPRT *xprt, xdrproc_t xdr_args, caddr_t args_ptr)
{
  return ((*xdr_args) (&(((struct tcp_conn *)
			  (xprt->xp_p1))->xdrs), args_ptr));
}

static bool_t
svctcp_freeargs (SVCXPRT *xprt, xdrproc_t xdr_args, caddr_t args_ptr)
{
  XDR *xdrs = &(((struct tcp_conn *) (xprt->xp_p1))->xdrs);

  xdrs->x_op = XDR_FREE;
  return ((*xdr_args) (xdrs, args_ptr));
}

static bool_t
svctcp_reply (SVCXPRT *xprt, struct rpc_msg *msg)
{
  struct tcp_conn *cd = (struct tcp_conn *) (xprt->xp_p1);
  XDR *xdrs = &(cd->xdrs);
  bool_t stat;

  xdrs->x_op = XDR_ENCODE;
  msg->rm_xid = cd->x_id;
  stat = xdr_replymsg (xdrs, msg);
  (void) xdrrec_endofrecord (xdrs, TRUE);
  return stat;
}
