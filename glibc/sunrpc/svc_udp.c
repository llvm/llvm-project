/*
 * svc_udp.c,
 * Server side for UDP/IP based RPC.  (Does some caching in the hopes of
 * achieving execute-at-most-once semantics.)
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
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <rpc/rpc.h>
#include <sys/socket.h>
#include <errno.h>
#include <libintl.h>

#ifdef IP_PKTINFO
#include <sys/uio.h>
#endif

#include <wchar.h>
#include <libio/iolibio.h>
#include <shlib-compat.h>

#define rpc_buffer(xprt) ((xprt)->xp_p1)
#ifndef MAX
#define MAX(a, b)     ((a > b) ? a : b)
#endif

static bool_t svcudp_recv (SVCXPRT *, struct rpc_msg *);
static bool_t svcudp_reply (SVCXPRT *, struct rpc_msg *);
static enum xprt_stat svcudp_stat (SVCXPRT *);
static bool_t svcudp_getargs (SVCXPRT *, xdrproc_t, caddr_t);
static bool_t svcudp_freeargs (SVCXPRT *, xdrproc_t, caddr_t);
static void svcudp_destroy (SVCXPRT *);

static const struct xp_ops svcudp_op =
{
  svcudp_recv,
  svcudp_stat,
  svcudp_getargs,
  svcudp_reply,
  svcudp_freeargs,
  svcudp_destroy
};

static int cache_get (SVCXPRT *, struct rpc_msg *, char **replyp,
		      u_long *replylenp);
static void cache_set (SVCXPRT *xprt, u_long replylen);

/*
 * kept in xprt->xp_p2
 */
struct svcudp_data
  {
    u_int su_iosz;		/* byte size of send.recv buffer */
    u_long su_xid;		/* transaction id */
    XDR su_xdrs;		/* XDR handle */
    char su_verfbody[MAX_AUTH_BYTES];	/* verifier body */
    char *su_cache;		/* cached data, NULL if no cache */
  };
#define	su_data(xprt)	((struct svcudp_data *)(xprt->xp_p2))

/*
 * Usage:
 *      xprt = svcudp_create(sock);
 *
 * If sock<0 then a socket is created, else sock is used.
 * If the socket, sock is not bound to a port then svcudp_create
 * binds it to an arbitrary port.  In any (successful) case,
 * xprt->xp_sock is the registered socket number and xprt->xp_port is the
 * associated port number.
 * Once *xprt is initialized, it is registered as a transporter;
 * see (svc.h, xprt_register).
 * The routines returns NULL if a problem occurred.
 */
SVCXPRT *
svcudp_bufcreate (int sock, u_int sendsz, u_int recvsz)
{
  bool_t madesock = FALSE;
  SVCXPRT *xprt;
  struct svcudp_data *su;
  struct sockaddr_in addr;
  socklen_t len = sizeof (struct sockaddr_in);
  int pad;
  void *buf;

  if (sock == RPC_ANYSOCK)
    {
      if ((sock = __socket (AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
	{
	  perror (_("svcudp_create: socket creation problem"));
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
  if (__getsockname (sock, (struct sockaddr *) &addr, &len) != 0)
    {
      perror (_("svcudp_create - cannot getsockname"));
      if (madesock)
	(void) __close (sock);
      return (SVCXPRT *) NULL;
    }
  xprt = (SVCXPRT *) mem_alloc (sizeof (SVCXPRT));
  su = (struct svcudp_data *) mem_alloc (sizeof (*su));
  buf = mem_alloc (((MAX (sendsz, recvsz) + 3) / 4) * 4);
  if (xprt == NULL || su == NULL || buf == NULL)
    {
      (void) __fxprintf (NULL, "%s: %s",
			 "svcudp_create",  _("out of memory\n"));
      mem_free (xprt, sizeof (SVCXPRT));
      mem_free (su, sizeof (*su));
      mem_free (buf, ((MAX (sendsz, recvsz) + 3) / 4) * 4);
      return NULL;
    }
  su->su_iosz = ((MAX (sendsz, recvsz) + 3) / 4) * 4;
  rpc_buffer (xprt) = buf;
  xdrmem_create (&(su->su_xdrs), rpc_buffer (xprt), su->su_iosz, XDR_DECODE);
  su->su_cache = NULL;
  xprt->xp_p2 = (caddr_t) su;
  xprt->xp_verf.oa_base = su->su_verfbody;
  xprt->xp_ops = &svcudp_op;
  xprt->xp_port = ntohs (addr.sin_port);
  xprt->xp_sock = sock;

#ifdef IP_PKTINFO
  if ((sizeof (struct iovec) + sizeof (struct msghdr)
       + sizeof(struct cmsghdr) + sizeof (struct in_pktinfo))
      > sizeof (xprt->xp_pad))
    {
      (void) __fxprintf (NULL,"%s", _("\
svcudp_create: xp_pad is too small for IP_PKTINFO\n"));
      return NULL;
    }
  pad = 1;
  if (__setsockopt (sock, SOL_IP, IP_PKTINFO, (void *) &pad,
		    sizeof (pad)) == 0)
    /* Set the padding to all 1s. */
    pad = 0xff;
  else
#endif
    /* Clear the padding. */
    pad = 0;
  memset (&xprt->xp_pad [0], pad, sizeof (xprt->xp_pad));

  xprt_register (xprt);
  return xprt;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svcudp_bufcreate)
#else
libc_hidden_nolink_sunrpc (svcudp_bufcreate, GLIBC_2_0)
#endif

SVCXPRT *
svcudp_create (int sock)
{
  return svcudp_bufcreate (sock, UDPMSGSIZE, UDPMSGSIZE);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svcudp_create)
#else
libc_hidden_nolink_sunrpc (svcudp_create, GLIBC_2_0)
#endif

static enum xprt_stat
svcudp_stat (SVCXPRT *xprt)
{

  return XPRT_IDLE;
}

static bool_t
svcudp_recv (SVCXPRT *xprt, struct rpc_msg *msg)
{
  struct svcudp_data *su = su_data (xprt);
  XDR *xdrs = &(su->su_xdrs);
  int rlen;
  char *reply;
  u_long replylen;
  socklen_t len;

  /* It is very tricky when you have IP aliases. We want to make sure
     that we are sending the packet from the IP address where the
     incoming packet is addressed to. H.J. */
#ifdef IP_PKTINFO
  struct iovec *iovp;
  struct msghdr *mesgp;
#endif

again:
  /* FIXME -- should xp_addrlen be a size_t?  */
  len = (socklen_t) sizeof(struct sockaddr_in);
#ifdef IP_PKTINFO
  iovp = (struct iovec *) &xprt->xp_pad [0];
  mesgp = (struct msghdr *) &xprt->xp_pad [sizeof (struct iovec)];
  if (mesgp->msg_iovlen)
    {
      iovp->iov_base = rpc_buffer (xprt);
      iovp->iov_len = su->su_iosz;
      mesgp->msg_iov = iovp;
      mesgp->msg_iovlen = 1;
      mesgp->msg_name = &(xprt->xp_raddr);
      mesgp->msg_namelen = len;
      mesgp->msg_control = &xprt->xp_pad [sizeof (struct iovec)
					  + sizeof (struct msghdr)];
      mesgp->msg_controllen = sizeof(xprt->xp_pad)
			      - sizeof (struct iovec) - sizeof (struct msghdr);
      rlen = __recvmsg (xprt->xp_sock, mesgp, 0);
      if (rlen >= 0)
	{
	  struct cmsghdr *cmsg;
	  len = mesgp->msg_namelen;
	  cmsg = CMSG_FIRSTHDR (mesgp);
	  if (cmsg == NULL
	      || CMSG_NXTHDR (mesgp, cmsg) != NULL
	      || cmsg->cmsg_level != SOL_IP
	      || cmsg->cmsg_type != IP_PKTINFO
	      || cmsg->cmsg_len < (sizeof (struct cmsghdr)
				   + sizeof (struct in_pktinfo)))
	    {
	      /* Not a simple IP_PKTINFO, ignore it.  */
	      mesgp->msg_control = NULL;
	      mesgp->msg_controllen = 0;
	    }
	  else
	    {
	      /* It was a simple IP_PKTIFO as we expected, discard the
		 interface field.  */
	      struct in_pktinfo *pkti = (struct in_pktinfo *) CMSG_DATA (cmsg);
	      pkti->ipi_ifindex = 0;
	    }
	}
    }
  else
#endif
    rlen = __recvfrom (xprt->xp_sock, rpc_buffer (xprt),
		       (int) su->su_iosz, 0,
		       (struct sockaddr *) &(xprt->xp_raddr), &len);
  xprt->xp_addrlen = len;
  if (rlen == -1)
    {
      if (errno == EINTR)
	goto again;
      __svc_accept_failed ();
    }
  if (rlen < 16)		/* < 4 32-bit ints? */
    return FALSE;
  xdrs->x_op = XDR_DECODE;
  XDR_SETPOS (xdrs, 0);
  if (!xdr_callmsg (xdrs, msg))
    return FALSE;
  su->su_xid = msg->rm_xid;
  if (su->su_cache != NULL)
    {
      if (cache_get (xprt, msg, &reply, &replylen))
	{
#ifdef IP_PKTINFO
	  if (mesgp->msg_iovlen)
	    {
	      iovp->iov_base = reply;
	      iovp->iov_len = replylen;
	      (void) __sendmsg (xprt->xp_sock, mesgp, 0);
	    }
	  else
#endif
	    (void) __sendto (xprt->xp_sock, reply, (int) replylen, 0,
			     (struct sockaddr *) &xprt->xp_raddr, len);
	  return TRUE;
	}
    }
  return TRUE;
}

static bool_t
svcudp_reply (SVCXPRT *xprt, struct rpc_msg *msg)
{
  struct svcudp_data *su = su_data (xprt);
  XDR *xdrs = &(su->su_xdrs);
  int slen, sent;
  bool_t stat = FALSE;
#ifdef IP_PKTINFO
  struct iovec *iovp;
  struct msghdr *mesgp;
#endif

  xdrs->x_op = XDR_ENCODE;
  XDR_SETPOS (xdrs, 0);
  msg->rm_xid = su->su_xid;
  if (xdr_replymsg (xdrs, msg))
    {
      slen = (int) XDR_GETPOS (xdrs);
#ifdef IP_PKTINFO
      mesgp = (struct msghdr *) &xprt->xp_pad [sizeof (struct iovec)];
      if (mesgp->msg_iovlen)
	{
	  iovp = (struct iovec *) &xprt->xp_pad [0];
	  iovp->iov_base = rpc_buffer (xprt);
	  iovp->iov_len = slen;
	  sent = __sendmsg (xprt->xp_sock, mesgp, 0);
	}
      else
#endif
	sent = __sendto (xprt->xp_sock, rpc_buffer (xprt), slen, 0,
			 (struct sockaddr *) &(xprt->xp_raddr),
			 xprt->xp_addrlen);
      if (sent == slen)
	{
	  stat = TRUE;
	  if (su->su_cache && slen >= 0)
	    {
	      cache_set (xprt, (u_long) slen);
	    }
	}
    }
  return stat;
}

static bool_t
svcudp_getargs (SVCXPRT *xprt, xdrproc_t xdr_args, caddr_t args_ptr)
{

  return (*xdr_args) (&(su_data (xprt)->su_xdrs), args_ptr);
}

static bool_t
svcudp_freeargs (SVCXPRT *xprt, xdrproc_t xdr_args, caddr_t args_ptr)
{
  XDR *xdrs = &(su_data (xprt)->su_xdrs);

  xdrs->x_op = XDR_FREE;
  return (*xdr_args) (xdrs, args_ptr);
}

static void
svcudp_destroy (SVCXPRT *xprt)
{
  struct svcudp_data *su = su_data (xprt);

  xprt_unregister (xprt);
  (void) __close (xprt->xp_sock);
  XDR_DESTROY (&(su->su_xdrs));
  mem_free (rpc_buffer (xprt), su->su_iosz);
  mem_free ((caddr_t) su, sizeof (struct svcudp_data));
  mem_free ((caddr_t) xprt, sizeof (SVCXPRT));
}


/***********this could be a separate file*********************/

/*
 * Fifo cache for udp server
 * Copies pointers to reply buffers into fifo cache
 * Buffers are sent again if retransmissions are detected.
 */

#define SPARSENESS 4		/* 75% sparse */

#define CACHE_PERROR(msg)	\
	(void) __fxprintf(NULL, "%s\n", msg)

#define ALLOC(type, size)	\
	(type *) mem_alloc((unsigned) (sizeof(type) * (size)))

#define CALLOC(type, size)	\
  (type *) calloc (sizeof (type), size)

/*
 * An entry in the cache
 */
typedef struct cache_node *cache_ptr;
struct cache_node
  {
    /*
     * Index into cache is xid, proc, vers, prog and address
     */
    u_long cache_xid;
    u_long cache_proc;
    u_long cache_vers;
    u_long cache_prog;
    struct sockaddr_in cache_addr;
    /*
     * The cached reply and length
     */
    char *cache_reply;
    u_long cache_replylen;
    /*
     * Next node on the list, if there is a collision
     */
    cache_ptr cache_next;
  };



/*
 * The entire cache
 */
struct udp_cache
  {
    u_long uc_size;		/* size of cache */
    cache_ptr *uc_entries;	/* hash table of entries in cache */
    cache_ptr *uc_fifo;		/* fifo list of entries in cache */
    u_long uc_nextvictim;	/* points to next victim in fifo list */
    u_long uc_prog;		/* saved program number */
    u_long uc_vers;		/* saved version number */
    u_long uc_proc;		/* saved procedure number */
    struct sockaddr_in uc_addr;	/* saved caller's address */
  };


/*
 * the hashing function
 */
#define CACHE_LOC(transp, xid)	\
 (xid % (SPARSENESS*((struct udp_cache *) su_data(transp)->su_cache)->uc_size))


/*
 * Enable use of the cache.
 * Note: there is no disable.
 */
int
svcudp_enablecache (SVCXPRT *transp, u_long size)
{
  struct svcudp_data *su = su_data (transp);
  struct udp_cache *uc;

  if (su->su_cache != NULL)
    {
      CACHE_PERROR (_("enablecache: cache already enabled"));
      return 0;
    }
  uc = ALLOC (struct udp_cache, 1);
  if (uc == NULL)
    {
      CACHE_PERROR (_("enablecache: could not allocate cache"));
      return 0;
    }
  uc->uc_size = size;
  uc->uc_nextvictim = 0;
  uc->uc_entries = CALLOC (cache_ptr, size * SPARSENESS);
  if (uc->uc_entries == NULL)
    {
      mem_free (uc, sizeof (struct udp_cache));
      CACHE_PERROR (_("enablecache: could not allocate cache data"));
      return 0;
    }
  uc->uc_fifo = CALLOC (cache_ptr, size);
  if (uc->uc_fifo == NULL)
    {
      mem_free (uc->uc_entries, size * SPARSENESS);
      mem_free (uc, sizeof (struct udp_cache));
      CACHE_PERROR (_("enablecache: could not allocate cache fifo"));
      return 0;
    }
  su->su_cache = (char *) uc;
  return 1;
}
libc_hidden_nolink_sunrpc (svcudp_enablecache, GLIBC_2_0)


/*
 * Set an entry in the cache
 */
static void
cache_set (SVCXPRT *xprt, u_long replylen)
{
  cache_ptr victim;
  cache_ptr *vicp;
  struct svcudp_data *su = su_data (xprt);
  struct udp_cache *uc = (struct udp_cache *) su->su_cache;
  u_int loc;
  char *newbuf;

  /*
   * Find space for the new entry, either by
   * reusing an old entry, or by mallocing a new one
   */
  victim = uc->uc_fifo[uc->uc_nextvictim];
  if (victim != NULL)
    {
      loc = CACHE_LOC (xprt, victim->cache_xid);
      for (vicp = &uc->uc_entries[loc];
	   *vicp != NULL && *vicp != victim;
	   vicp = &(*vicp)->cache_next)
	;
      if (*vicp == NULL)
	{
	  CACHE_PERROR (_("cache_set: victim not found"));
	  return;
	}
      *vicp = victim->cache_next;	/* remote from cache */
      newbuf = victim->cache_reply;
    }
  else
    {
      victim = ALLOC (struct cache_node, 1);
      if (victim == NULL)
	{
	  CACHE_PERROR (_("cache_set: victim alloc failed"));
	  return;
	}
      newbuf = mem_alloc (su->su_iosz);
      if (newbuf == NULL)
	{
	  mem_free (victim, sizeof (struct cache_node));
	  CACHE_PERROR (_("cache_set: could not allocate new rpc_buffer"));
	  return;
	}
    }

  /*
   * Store it away
   */
  victim->cache_replylen = replylen;
  victim->cache_reply = rpc_buffer (xprt);
  rpc_buffer (xprt) = newbuf;
  xdrmem_create (&(su->su_xdrs), rpc_buffer (xprt), su->su_iosz, XDR_ENCODE);
  victim->cache_xid = su->su_xid;
  victim->cache_proc = uc->uc_proc;
  victim->cache_vers = uc->uc_vers;
  victim->cache_prog = uc->uc_prog;
  victim->cache_addr = uc->uc_addr;
  loc = CACHE_LOC (xprt, victim->cache_xid);
  victim->cache_next = uc->uc_entries[loc];
  uc->uc_entries[loc] = victim;
  uc->uc_fifo[uc->uc_nextvictim++] = victim;
  uc->uc_nextvictim %= uc->uc_size;
}

/*
 * Try to get an entry from the cache
 * return 1 if found, 0 if not found
 */
static int
cache_get (SVCXPRT *xprt, struct rpc_msg *msg, char **replyp,
	   u_long *replylenp)
{
  u_int loc;
  cache_ptr ent;
  struct svcudp_data *su = su_data (xprt);
  struct udp_cache *uc = (struct udp_cache *) su->su_cache;

#define EQADDR(a1, a2)	(memcmp((char*)&a1, (char*)&a2, sizeof(a1)) == 0)

  loc = CACHE_LOC (xprt, su->su_xid);
  for (ent = uc->uc_entries[loc]; ent != NULL; ent = ent->cache_next)
    {
      if (ent->cache_xid == su->su_xid &&
	  ent->cache_proc == uc->uc_proc &&
	  ent->cache_vers == uc->uc_vers &&
	  ent->cache_prog == uc->uc_prog &&
	  EQADDR (ent->cache_addr, uc->uc_addr))
	{
	  *replyp = ent->cache_reply;
	  *replylenp = ent->cache_replylen;
	  return 1;
	}
    }
  /*
   * Failed to find entry
   * Remember a few things so we can do a set later
   */
  uc->uc_proc = msg->rm_call.cb_proc;
  uc->uc_vers = msg->rm_call.cb_vers;
  uc->uc_prog = msg->rm_call.cb_prog;
  memcpy (&uc->uc_addr, &xprt->xp_raddr, sizeof (uc->uc_addr));
  return 0;
}
