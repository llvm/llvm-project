/*
 * svc.c, Server-side remote procedure call interface.
 *
 * There are two sets of procedures here.  The xprt routines are
 * for handling transport handles.  The svc routines handle the
 * list of service routines.
 *  Copyright (C) 2002-2021 Free Software Foundation, Inc.
 *  This file is part of the GNU C Library.
 *  Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.
 *
 *  The GNU C Library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  The GNU C Library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with the GNU C Library; if not, see
 *  <https://www.gnu.org/licenses/>.
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

#include <errno.h>
#include <unistd.h>
#include <rpc/rpc.h>
#include <rpc/svc.h>
#include <rpc/pmap_clnt.h>
#include <sys/poll.h>
#include <time.h>
#include <shlib-compat.h>

#define xports RPC_THREAD_VARIABLE(svc_xports_s)

#define NULL_SVC ((struct svc_callout *)0)
#define	RQCRED_SIZE	400	/* this size is excessive */

/* The services list
   Each entry represents a set of procedures (an rpc program).
   The dispatch routine takes request structs and runs the
   appropriate procedure. */
struct svc_callout {
  struct svc_callout *sc_next;
  rpcprog_t sc_prog;
  rpcvers_t sc_vers;
  void (*sc_dispatch) (struct svc_req *, SVCXPRT *);
  bool_t sc_mapped;
};
#define svc_head RPC_THREAD_VARIABLE(svc_head_s)

/* ***************  SVCXPRT related stuff **************** */

/* Activate a transport handle. */
void
xprt_register (SVCXPRT *xprt)
{
  register int sock = xprt->xp_sock;
  register int i;

  if (xports == NULL)
    {
      xports = (SVCXPRT **) calloc (_rpc_dtablesize (), sizeof (SVCXPRT *));
      if (xports == NULL) /* Don't add handle */
	return;
    }

  if (sock < _rpc_dtablesize ())
    {
      struct pollfd *new_svc_pollfd;

      xports[sock] = xprt;
      if (sock < FD_SETSIZE)
	FD_SET (sock, &svc_fdset);

      /* Check if we have an empty slot */
      for (i = 0; i < svc_max_pollfd; ++i)
	if (svc_pollfd[i].fd == -1)
	  {
	    svc_pollfd[i].fd = sock;
	    svc_pollfd[i].events = (POLLIN | POLLPRI |
				    POLLRDNORM | POLLRDBAND);
	    return;
	  }

      new_svc_pollfd = (struct pollfd *) realloc (svc_pollfd,
						  sizeof (struct pollfd)
						  * (svc_max_pollfd + 1));
      if (new_svc_pollfd == NULL) /* Out of memory */
	return;
      svc_pollfd = new_svc_pollfd;
      ++svc_max_pollfd;

      svc_pollfd[svc_max_pollfd - 1].fd = sock;
      svc_pollfd[svc_max_pollfd - 1].events = (POLLIN | POLLPRI |
					       POLLRDNORM | POLLRDBAND);
    }
}
libc_hidden_nolink_sunrpc (xprt_register, GLIBC_2_0)

/* De-activate a transport handle. */
void
xprt_unregister (SVCXPRT *xprt)
{
  register int sock = xprt->xp_sock;
  register int i;

  if ((sock < _rpc_dtablesize ()) && (xports[sock] == xprt))
    {
      xports[sock] = (SVCXPRT *) 0;

      if (sock < FD_SETSIZE)
	FD_CLR (sock, &svc_fdset);

      for (i = 0; i < svc_max_pollfd; ++i)
	if (svc_pollfd[i].fd == sock)
	  svc_pollfd[i].fd = -1;
    }
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (xprt_unregister)
#else
libc_hidden_nolink_sunrpc (xprt_unregister, GLIBC_2_0)
#endif


/* ********************** CALLOUT list related stuff ************* */

/* Search the callout list for a program number, return the callout
   struct. */
static struct svc_callout *
svc_find (rpcprog_t prog, rpcvers_t vers, struct svc_callout **prev)
{
  register struct svc_callout *s, *p;

  p = NULL_SVC;
  for (s = svc_head; s != NULL_SVC; s = s->sc_next)
    {
      if ((s->sc_prog == prog) && (s->sc_vers == vers))
	goto done;
      p = s;
    }
done:
  *prev = p;
  return s;
}

/* Add a service program to the callout list.
   The dispatch routine will be called when a rpc request for this
   program number comes in. */
bool_t
svc_register (SVCXPRT * xprt, rpcprog_t prog, rpcvers_t vers,
	      void (*dispatch) (struct svc_req *, SVCXPRT *),
	      rpcproc_t protocol)
{
  struct svc_callout *prev;
  register struct svc_callout *s;

  if ((s = svc_find (prog, vers, &prev)) != NULL_SVC)
    {
      if (s->sc_dispatch == dispatch)
	goto pmap_it;		/* he is registering another xptr */
      return FALSE;
    }
  s = (struct svc_callout *) mem_alloc (sizeof (struct svc_callout));
  if (s == (struct svc_callout *) 0)
    return FALSE;

  s->sc_prog = prog;
  s->sc_vers = vers;
  s->sc_dispatch = dispatch;
  s->sc_next = svc_head;
  s->sc_mapped = FALSE;
  svc_head = s;

pmap_it:
  /* now register the information with the local binder service */
  if (protocol)
    {
      if (! pmap_set (prog, vers, protocol, xprt->xp_port))
	return FALSE;

      s->sc_mapped = TRUE;
    }

  return TRUE;
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svc_register)
#else
libc_hidden_nolink_sunrpc (svc_register, GLIBC_2_0)
#endif

/* Remove a service program from the callout list. */
void
svc_unregister (rpcprog_t prog, rpcvers_t vers)
{
  struct svc_callout *prev;
  register struct svc_callout *s;

  if ((s = svc_find (prog, vers, &prev)) == NULL_SVC)
    return;
  bool is_mapped = s->sc_mapped;

  if (prev == NULL_SVC)
    svc_head = s->sc_next;
  else
    prev->sc_next = s->sc_next;

  s->sc_next = NULL_SVC;
  mem_free ((char *) s, (u_int) sizeof (struct svc_callout));
  /* now unregister the information with the local binder service */
  if (is_mapped)
    pmap_unset (prog, vers);
}
libc_hidden_nolink_sunrpc (svc_unregister, GLIBC_2_0)

/* ******************* REPLY GENERATION ROUTINES  ************ */

/* Send a reply to an rpc request */
bool_t
svc_sendreply (register SVCXPRT *xprt, xdrproc_t xdr_results,
	       caddr_t xdr_location)
{
  struct rpc_msg rply;

  rply.rm_direction = REPLY;
  rply.rm_reply.rp_stat = MSG_ACCEPTED;
  rply.acpted_rply.ar_verf = xprt->xp_verf;
  rply.acpted_rply.ar_stat = SUCCESS;
  rply.acpted_rply.ar_results.where = xdr_location;
  rply.acpted_rply.ar_results.proc = xdr_results;
  return SVC_REPLY (xprt, &rply);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svc_sendreply)
#else
libc_hidden_nolink_sunrpc (svc_sendreply, GLIBC_2_0)
#endif

/* No procedure error reply */
void
svcerr_noproc (register SVCXPRT *xprt)
{
  struct rpc_msg rply;

  rply.rm_direction = REPLY;
  rply.rm_reply.rp_stat = MSG_ACCEPTED;
  rply.acpted_rply.ar_verf = xprt->xp_verf;
  rply.acpted_rply.ar_stat = PROC_UNAVAIL;
  SVC_REPLY (xprt, &rply);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svcerr_noproc)
#else
libc_hidden_nolink_sunrpc (svcerr_noproc, GLIBC_2_0)
#endif

/* Can't decode args error reply */
void
svcerr_decode (register SVCXPRT *xprt)
{
  struct rpc_msg rply;

  rply.rm_direction = REPLY;
  rply.rm_reply.rp_stat = MSG_ACCEPTED;
  rply.acpted_rply.ar_verf = xprt->xp_verf;
  rply.acpted_rply.ar_stat = GARBAGE_ARGS;
  SVC_REPLY (xprt, &rply);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svcerr_decode)
#else
libc_hidden_nolink_sunrpc (svcerr_decode, GLIBC_2_0)
#endif

/* Some system error */
void
svcerr_systemerr (register SVCXPRT *xprt)
{
  struct rpc_msg rply;

  rply.rm_direction = REPLY;
  rply.rm_reply.rp_stat = MSG_ACCEPTED;
  rply.acpted_rply.ar_verf = xprt->xp_verf;
  rply.acpted_rply.ar_stat = SYSTEM_ERR;
  SVC_REPLY (xprt, &rply);
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svcerr_systemerr)
#else
libc_hidden_nolink_sunrpc (svcerr_systemerr, GLIBC_2_0)
#endif

/* Authentication error reply */
void
svcerr_auth (SVCXPRT *xprt, enum auth_stat why)
{
  struct rpc_msg rply;

  rply.rm_direction = REPLY;
  rply.rm_reply.rp_stat = MSG_DENIED;
  rply.rjcted_rply.rj_stat = AUTH_ERROR;
  rply.rjcted_rply.rj_why = why;
  SVC_REPLY (xprt, &rply);
}
libc_hidden_nolink_sunrpc (svcerr_auth, GLIBC_2_0)

/* Auth too weak error reply */
void
svcerr_weakauth (SVCXPRT *xprt)
{
  svcerr_auth (xprt, AUTH_TOOWEAK);
}
libc_hidden_nolink_sunrpc (svcerr_weakauth, GLIBC_2_0)

/* Program unavailable error reply */
void
svcerr_noprog (register SVCXPRT *xprt)
{
  struct rpc_msg rply;

  rply.rm_direction = REPLY;
  rply.rm_reply.rp_stat = MSG_ACCEPTED;
  rply.acpted_rply.ar_verf = xprt->xp_verf;
  rply.acpted_rply.ar_stat = PROG_UNAVAIL;
  SVC_REPLY (xprt, &rply);
}
libc_hidden_nolink_sunrpc (svcerr_noprog, GLIBC_2_0)

/* Program version mismatch error reply */
void
svcerr_progvers (register SVCXPRT *xprt, rpcvers_t low_vers,
		 rpcvers_t high_vers)
{
  struct rpc_msg rply;

  rply.rm_direction = REPLY;
  rply.rm_reply.rp_stat = MSG_ACCEPTED;
  rply.acpted_rply.ar_verf = xprt->xp_verf;
  rply.acpted_rply.ar_stat = PROG_MISMATCH;
  rply.acpted_rply.ar_vers.low = low_vers;
  rply.acpted_rply.ar_vers.high = high_vers;
  SVC_REPLY (xprt, &rply);
}
libc_hidden_nolink_sunrpc (svcerr_progvers, GLIBC_2_0)

/* ******************* SERVER INPUT STUFF ******************* */

/*
 * Get server side input from some transport.
 *
 * Statement of authentication parameters management:
 * This function owns and manages all authentication parameters, specifically
 * the "raw" parameters (msg.rm_call.cb_cred and msg.rm_call.cb_verf) and
 * the "cooked" credentials (rqst->rq_clntcred).
 * However, this function does not know the structure of the cooked
 * credentials, so it make the following assumptions:
 *   a) the structure is contiguous (no pointers), and
 *   b) the cred structure size does not exceed RQCRED_SIZE bytes.
 * In all events, all three parameters are freed upon exit from this routine.
 * The storage is trivially management on the call stack in user land, but
 * is mallocated in kernel land.
 */

void
svc_getreq (int rdfds)
{
  fd_set readfds;

  FD_ZERO (&readfds);
  readfds.fds_bits[0] = rdfds;
  svc_getreqset (&readfds);
}
libc_hidden_nolink_sunrpc (svc_getreq, GLIBC_2_0)

void
svc_getreqset (fd_set *readfds)
{
  register fd_mask mask;
  register fd_mask *maskp;
  register int setsize;
  register int sock;
  register int bit;

  setsize = _rpc_dtablesize ();
  if (setsize > FD_SETSIZE)
    setsize = FD_SETSIZE;
  maskp = readfds->fds_bits;
  for (sock = 0; sock < setsize; sock += NFDBITS)
    for (mask = *maskp++; (bit = ffsl (mask)); mask ^= (1L << (bit - 1)))
      svc_getreq_common (sock + bit - 1);
}
libc_hidden_nolink_sunrpc (svc_getreqset, GLIBC_2_0)

void
svc_getreq_poll (struct pollfd *pfdp, int pollretval)
{
  if (pollretval == 0)
    return;

  register int fds_found;
  for (int i = fds_found = 0; i < svc_max_pollfd; ++i)
    {
      register struct pollfd *p = &pfdp[i];

      if (p->fd != -1 && p->revents)
	{
	  /* fd has input waiting */
	  if (p->revents & POLLNVAL)
	    xprt_unregister (xports[p->fd]);
	  else
	    svc_getreq_common (p->fd);

	  if (++fds_found >= pollretval)
	    break;
	}
    }
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (svc_getreq_poll)
#else
libc_hidden_nolink_sunrpc (svc_getreq_poll, GLIBC_2_2)
#endif


void
svc_getreq_common (const int fd)
{
  enum xprt_stat stat;
  struct rpc_msg msg;
  register SVCXPRT *xprt;
  char cred_area[2 * MAX_AUTH_BYTES + RQCRED_SIZE];
  msg.rm_call.cb_cred.oa_base = cred_area;
  msg.rm_call.cb_verf.oa_base = &(cred_area[MAX_AUTH_BYTES]);

  xprt = xports[fd];
  /* Do we control fd? */
  if (xprt == NULL)
     return;

  /* now receive msgs from xprtprt (support batch calls) */
  do
    {
      if (SVC_RECV (xprt, &msg))
	{
	  /* now find the exported program and call it */
	  struct svc_callout *s;
	  struct svc_req r;
	  enum auth_stat why;
	  rpcvers_t low_vers;
	  rpcvers_t high_vers;
	  int prog_found;

	  r.rq_clntcred = &(cred_area[2 * MAX_AUTH_BYTES]);
	  r.rq_xprt = xprt;
	  r.rq_prog = msg.rm_call.cb_prog;
	  r.rq_vers = msg.rm_call.cb_vers;
	  r.rq_proc = msg.rm_call.cb_proc;
	  r.rq_cred = msg.rm_call.cb_cred;

	  /* first authenticate the message */
	  /* Check for null flavor and bypass these calls if possible */

	  if (msg.rm_call.cb_cred.oa_flavor == AUTH_NULL)
	    {
	      r.rq_xprt->xp_verf.oa_flavor = _null_auth.oa_flavor;
	      r.rq_xprt->xp_verf.oa_length = 0;
	    }
	  else if ((why = _authenticate (&r, &msg)) != AUTH_OK)
	    {
	      svcerr_auth (xprt, why);
	      goto call_done;
	    }

	  /* now match message with a registered service */
	  prog_found = FALSE;
	  low_vers = 0 - 1;
	  high_vers = 0;

	  for (s = svc_head; s != NULL_SVC; s = s->sc_next)
	    {
	      if (s->sc_prog == r.rq_prog)
		{
		  if (s->sc_vers == r.rq_vers)
		    {
		      (*s->sc_dispatch) (&r, xprt);
		      goto call_done;
		    }
		  /* found correct version */
		  prog_found = TRUE;
		  if (s->sc_vers < low_vers)
		    low_vers = s->sc_vers;
		  if (s->sc_vers > high_vers)
		    high_vers = s->sc_vers;
		}
	      /* found correct program */
	    }
	  /* if we got here, the program or version
	     is not served ... */
	  if (prog_found)
	    svcerr_progvers (xprt, low_vers, high_vers);
	  else
	    svcerr_noprog (xprt);
	  /* Fall through to ... */
	}
    call_done:
      if ((stat = SVC_STAT (xprt)) == XPRT_DIED)
	{
	  SVC_DESTROY (xprt);
	  break;
	}
    }
  while (stat == XPRT_MOREREQS);
}
libc_hidden_nolink_sunrpc (svc_getreq_common, GLIBC_2_2)

void
__svc_wait_on_error (void)
{
  struct timespec ts = { .tv_sec = 0, .tv_nsec = 50000000 };
  __nanosleep (&ts, NULL);
}

/* If there are no file descriptors available, then accept will fail.
   We want to delay here so the connection request can be dequeued;
   otherwise we can bounce between polling and accepting, never giving the
   request a chance to dequeue and eating an enormous amount of cpu time
   in svc_run if we're polling on many file descriptors.  */
void
__svc_accept_failed (void)
{
  if (errno == EMFILE)
    {
      __svc_wait_on_error ();
    }
}


void
__rpc_thread_svc_cleanup (void)
{
  struct svc_callout *svcp;

  while ((svcp = svc_head) != NULL)
    svc_unregister (svcp->sc_prog, svcp->sc_vers);
}
