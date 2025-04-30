/* Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

/*
 * Copyright (c) 1985, 1989, 1993
 *    The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Portions Copyright (c) 1993 by Digital Equipment Corporation.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies, and that
 * the name of Digital Equipment Corporation not be used in advertising or
 * publicity pertaining to distribution of the document or software without
 * specific, written prior permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND DIGITAL EQUIPMENT CORP. DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS.   IN NO EVENT SHALL DIGITAL EQUIPMENT
 * CORPORATION BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 */

/*
 * Portions Copyright (c) 1996-1999 by Internet Software Consortium.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND INTERNET SOFTWARE CONSORTIUM DISCLAIMS
 * ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL INTERNET SOFTWARE
 * CONSORTIUM BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 */

/*
 * Send query to name server and wait for reply.
 */

#include <assert.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/poll.h>

#include <netinet/in.h>
#include <arpa/nameser.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <kernel-features.h>
#include <libc-diag.h>
#include <random-bits.h>

#if PACKETSZ > 65536
#define MAXPACKET       PACKETSZ
#else
#define MAXPACKET       65536
#endif

/* From ev_streams.c.  */

static inline void
__attribute ((always_inline))
evConsIovec(void *buf, size_t cnt, struct iovec *vec) {
	memset(vec, 0xf5, sizeof (*vec));
	vec->iov_base = buf;
	vec->iov_len = cnt;
}

/* From ev_timers.c.  */

#define BILLION 1000000000

static inline void
evConsTime(struct timespec *res, time_t sec, long nsec) {
	res->tv_sec = sec;
	res->tv_nsec = nsec;
}

static inline void
evAddTime(struct timespec *res, const struct timespec *addend1,
	  const struct timespec *addend2) {
	res->tv_sec = addend1->tv_sec + addend2->tv_sec;
	res->tv_nsec = addend1->tv_nsec + addend2->tv_nsec;
	if (res->tv_nsec >= BILLION) {
		res->tv_sec++;
		res->tv_nsec -= BILLION;
	}
}

static inline void
evSubTime(struct timespec *res, const struct timespec *minuend,
	  const struct timespec *subtrahend) {
       res->tv_sec = minuend->tv_sec - subtrahend->tv_sec;
	if (minuend->tv_nsec >= subtrahend->tv_nsec)
		res->tv_nsec = minuend->tv_nsec - subtrahend->tv_nsec;
	else {
		res->tv_nsec = (BILLION
				- subtrahend->tv_nsec + minuend->tv_nsec);
		res->tv_sec--;
	}
}

static int
evCmpTime(struct timespec a, struct timespec b) {
	long x = a.tv_sec - b.tv_sec;

	if (x == 0L)
		x = a.tv_nsec - b.tv_nsec;
	return (x < 0L ? (-1) : x > 0L ? (1) : (0));
}

static void
evNowTime(struct timespec *res) {
	__clock_gettime(CLOCK_REALTIME, res);
}


#define EXT(res) ((res)->_u._ext)

/* Forward. */

static int		send_vc(res_state, const u_char *, int,
				const u_char *, int,
				u_char **, int *, int *, int, u_char **,
				u_char **, int *, int *, int *);
static int		send_dg(res_state, const u_char *, int,
				const u_char *, int,
				u_char **, int *, int *, int,
				int *, int *, u_char **,
				u_char **, int *, int *, int *);
static int		sock_eq(struct sockaddr_in6 *, struct sockaddr_in6 *);

/* Returns a shift value for the name server index.  Used to implement
   RES_ROTATE.  */
static unsigned int
nameserver_offset (struct __res_state *statp)
{
  /* If we only have one name server or rotation is disabled, return
     offset 0 (no rotation).  */
  unsigned int nscount = statp->nscount;
  if (nscount <= 1 || !(statp->options & RES_ROTATE))
    return 0;

  /* Global offset.  The lowest bit indicates whether the offset has
     been initialized with a random value.  Use relaxed MO to access
     global_offset because all we need is a sequence of roughly
     sequential value.  */
  static unsigned int global_offset;
  unsigned int offset = atomic_fetch_add_relaxed (&global_offset, 2);
  if ((offset & 1) == 0)
    {
      /* Initialization is required.  */
      offset = random_bits ();
      /* The lowest bit is the most random.  Preserve it.  */
      offset <<= 1;

      /* Store the new starting value.  atomic_fetch_add_relaxed
	 returns the old value, so emulate that by storing the new
	 (incremented) value.  Concurrent initialization with
	 different random values is harmless.  */
      atomic_store_relaxed (&global_offset, (offset | 1) + 2);
    }

  /* Remove the initialization bit.  */
  offset >>= 1;

  /* Avoid the division in the most common cases.  */
  switch (nscount)
    {
    case 2:
      return offset & 1;
    case 3:
      return offset % 3;
    case 4:
      return offset & 3;
    default:
      return offset % nscount;
    }
}

/* Clear the AD bit unless the trust-ad option was specified in the
   resolver configuration.  */
static void
mask_ad_bit (struct resolv_context *ctx, void *buf)
{
  if (!(ctx->resp->options & RES_TRUSTAD))
    ((HEADER *) buf)->ad = 0;
}

int
__res_context_send (struct resolv_context *ctx,
		    const unsigned char *buf, int buflen,
		    const unsigned char *buf2, int buflen2,
		    unsigned char *ans, int anssiz,
		    unsigned char **ansp, unsigned char **ansp2,
		    int *nansp2, int *resplen2, int *ansp2_malloced)
{
	struct __res_state *statp = ctx->resp;
	int gotsomewhere, terrno, try, v_circuit, resplen;
	/* On some architectures send_vc is inlined and the compiler might emit
	   a warning indicating 'resplen' may be used uninitialized.  Note that
	   the warning belongs to resplen in send_vc which is used as return
	   value!  There the maybe-uninitialized warning is already ignored as
	   it is a false-positive - see comment in send_vc.
	   Here the variable n is set to the return value of send_vc.
	   See below.  */
	DIAG_PUSH_NEEDS_COMMENT;
	DIAG_IGNORE_NEEDS_COMMENT (9, "-Wmaybe-uninitialized");
	int n;
	DIAG_POP_NEEDS_COMMENT;

	if (statp->nscount == 0) {
		__set_errno (ESRCH);
		return (-1);
	}

	if (anssiz < (buf2 == NULL ? 1 : 2) * HFIXEDSZ) {
		__set_errno (EINVAL);
		return (-1);
	}

	v_circuit = ((statp->options & RES_USEVC)
		     || buflen > PACKETSZ
		     || buflen2 > PACKETSZ);
	gotsomewhere = 0;
	terrno = ETIMEDOUT;

	/*
	 * If the ns_addr_list in the resolver context has changed, then
	 * invalidate our cached copy and the associated timing data.
	 */
	if (EXT(statp).nscount != 0) {
		int needclose = 0;

		if (EXT(statp).nscount != statp->nscount)
			needclose++;
		else
			for (unsigned int ns = 0; ns < statp->nscount; ns++) {
				if (statp->nsaddr_list[ns].sin_family != 0
				    && !sock_eq((struct sockaddr_in6 *)
						&statp->nsaddr_list[ns],
						EXT(statp).nsaddrs[ns]))
				{
					needclose++;
					break;
				}
			}
		if (needclose) {
			__res_iclose(statp, false);
			EXT(statp).nscount = 0;
		}
	}

	/*
	 * Maybe initialize our private copy of the ns_addr_list.
	 */
	if (EXT(statp).nscount == 0) {
		for (unsigned int ns = 0; ns < statp->nscount; ns++) {
			EXT(statp).nssocks[ns] = -1;
			if (statp->nsaddr_list[ns].sin_family == 0)
				continue;
			if (EXT(statp).nsaddrs[ns] == NULL)
				EXT(statp).nsaddrs[ns] =
				    malloc(sizeof (struct sockaddr_in6));
			if (EXT(statp).nsaddrs[ns] != NULL)
				memset (mempcpy(EXT(statp).nsaddrs[ns],
						&statp->nsaddr_list[ns],
						sizeof (struct sockaddr_in)),
					'\0',
					sizeof (struct sockaddr_in6)
					- sizeof (struct sockaddr_in));
			else
				return -1;
		}
		EXT(statp).nscount = statp->nscount;
	}

	/* Name server index offset.  Used to implement
	   RES_ROTATE.  */
	unsigned int ns_offset = nameserver_offset (statp);

	/*
	 * Send request, RETRY times, or until successful.
	 */
	for (try = 0; try < statp->retry; try++) {
	    for (unsigned ns_shift = 0; ns_shift < statp->nscount; ns_shift++)
	    {
		/* The actual name server index.  This implements
		   RES_ROTATE.  */
		unsigned int ns = ns_shift + ns_offset;
		if (ns >= statp->nscount)
			ns -= statp->nscount;

	    same_ns:
		if (__glibc_unlikely (v_circuit))       {
			/* Use VC; at most one attempt per server. */
			try = statp->retry;
			n = send_vc(statp, buf, buflen, buf2, buflen2,
				    &ans, &anssiz, &terrno,
				    ns, ansp, ansp2, nansp2, resplen2,
				    ansp2_malloced);
			if (n < 0)
				return (-1);
			/* See comment at the declaration of n.  */
			DIAG_PUSH_NEEDS_COMMENT;
			DIAG_IGNORE_NEEDS_COMMENT (9, "-Wmaybe-uninitialized");
			if (n == 0 && (buf2 == NULL || *resplen2 == 0))
				goto next_ns;
			DIAG_POP_NEEDS_COMMENT;
		} else {
			/* Use datagrams. */
			n = send_dg(statp, buf, buflen, buf2, buflen2,
				    &ans, &anssiz, &terrno,
				    ns, &v_circuit, &gotsomewhere, ansp,
				    ansp2, nansp2, resplen2, ansp2_malloced);
			if (n < 0)
				return (-1);
			if (n == 0 && (buf2 == NULL || *resplen2 == 0))
				goto next_ns;
			if (v_circuit)
			  // XXX Check whether both requests failed or
			  // XXX whether one has been answered successfully
				goto same_ns;
		}

		resplen = n;

		/* See comment at the declaration of n.  Note: resplen = n;  */
		DIAG_PUSH_NEEDS_COMMENT;
		DIAG_IGNORE_NEEDS_COMMENT (9, "-Wmaybe-uninitialized");
		/* Mask the AD bit in both responses unless it is
		   marked trusted.  */
		if (resplen > HFIXEDSZ)
		  {
		    if (ansp != NULL)
		      mask_ad_bit (ctx, *ansp);
		    else
		      mask_ad_bit (ctx, ans);
		  }
		DIAG_POP_NEEDS_COMMENT;
		if (resplen2 != NULL && *resplen2 > HFIXEDSZ)
		  mask_ad_bit (ctx, *ansp2);

		/*
		 * If we have temporarily opened a virtual circuit,
		 * or if we haven't been asked to keep a socket open,
		 * close the socket.
		 */
		if ((v_circuit && (statp->options & RES_USEVC) == 0) ||
		    (statp->options & RES_STAYOPEN) == 0) {
			__res_iclose(statp, false);
		}
		return (resplen);
 next_ns: ;
	   } /*foreach ns*/
	} /*foreach retry*/
	__res_iclose(statp, false);
	if (!v_circuit) {
		if (!gotsomewhere)
			__set_errno (ECONNREFUSED);	/* no nameservers found */
		else
			__set_errno (ETIMEDOUT);	/* no answer obtained */
	} else
		__set_errno (terrno);
	return (-1);
}
libc_hidden_def (__res_context_send)

/* Common part of res_nsend and res_send.  */
static int
context_send_common (struct resolv_context *ctx,
		     const unsigned char *buf, int buflen,
		     unsigned char *ans, int anssiz)
{
  if (ctx == NULL)
    {
      RES_SET_H_ERRNO (&_res, NETDB_INTERNAL);
      return -1;
    }
  int result = __res_context_send (ctx, buf, buflen, NULL, 0, ans, anssiz,
				   NULL, NULL, NULL, NULL, NULL);
  __resolv_context_put (ctx);
  return result;
}

int
___res_nsend (res_state statp, const unsigned char *buf, int buflen,
	      unsigned char *ans, int anssiz)
{
  return context_send_common
    (__resolv_context_get_override (statp), buf, buflen, ans, anssiz);
}
versioned_symbol (libc, ___res_nsend, res_nsend, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_nsend, __res_nsend, GLIBC_2_2);
#endif

int
___res_send (const unsigned char *buf, int buflen, unsigned char *ans,
	     int anssiz)
{
  return context_send_common
    (__resolv_context_get (), buf, buflen, ans, anssiz);
}
versioned_symbol (libc, ___res_send, res_send, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libresolv, ___res_send, __res_send, GLIBC_2_0);
#endif

/* Private */

/* Close the resolver structure, assign zero to *RESPLEN2 if RESPLEN2
   is not NULL, and return zero.  */
static int
__attribute__ ((warn_unused_result))
close_and_return_error (res_state statp, int *resplen2)
{
  __res_iclose(statp, false);
  if (resplen2 != NULL)
    *resplen2 = 0;
  return 0;
}

/* The send_vc function is responsible for sending a DNS query over TCP
   to the nameserver numbered NS from the res_state STATP i.e.
   EXT(statp).nssocks[ns].  The function supports sending both IPv4 and
   IPv6 queries at the same serially on the same socket.

   Please note that for TCP there is no way to disable sending both
   queries, unlike UDP, which honours RES_SNGLKUP and RES_SNGLKUPREOP
   and sends the queries serially and waits for the result after each
   sent query.  This implementation should be corrected to honour these
   options.

   Please also note that for TCP we send both queries over the same
   socket one after another.  This technically violates best practice
   since the server is allowed to read the first query, respond, and
   then close the socket (to service another client).  If the server
   does this, then the remaining second query in the socket data buffer
   will cause the server to send the client an RST which will arrive
   asynchronously and the client's OS will likely tear down the socket
   receive buffer resulting in a potentially short read and lost
   response data.  This will force the client to retry the query again,
   and this process may repeat until all servers and connection resets
   are exhausted and then the query will fail.  It's not known if this
   happens with any frequency in real DNS server implementations.  This
   implementation should be corrected to use two sockets by default for
   parallel queries.

   The query stored in BUF of BUFLEN length is sent first followed by
   the query stored in BUF2 of BUFLEN2 length.  Queries are sent
   serially on the same socket.

   Answers to the query are stored firstly in *ANSP up to a max of
   *ANSSIZP bytes.  If more than *ANSSIZP bytes are needed and ANSCP
   is non-NULL (to indicate that modifying the answer buffer is allowed)
   then malloc is used to allocate a new response buffer and ANSCP and
   ANSP will both point to the new buffer.  If more than *ANSSIZP bytes
   are needed but ANSCP is NULL, then as much of the response as
   possible is read into the buffer, but the results will be truncated.
   When truncation happens because of a small answer buffer the DNS
   packets header field TC will bet set to 1, indicating a truncated
   message and the rest of the socket data will be read and discarded.

   Answers to the query are stored secondly in *ANSP2 up to a max of
   *ANSSIZP2 bytes, with the actual response length stored in
   *RESPLEN2.  If more than *ANSSIZP bytes are needed and ANSP2
   is non-NULL (required for a second query) then malloc is used to
   allocate a new response buffer, *ANSSIZP2 is set to the new buffer
   size and *ANSP2_MALLOCED is set to 1.

   The ANSP2_MALLOCED argument will eventually be removed as the
   change in buffer pointer can be used to detect the buffer has
   changed and that the caller should use free on the new buffer.

   Note that the answers may arrive in any order from the server and
   therefore the first and second answer buffers may not correspond to
   the first and second queries.

   It is not supported to call this function with a non-NULL ANSP2
   but a NULL ANSCP.  Put another way, you can call send_vc with a
   single unmodifiable buffer or two modifiable buffers, but no other
   combination is supported.

   It is the caller's responsibility to free the malloc allocated
   buffers by detecting that the pointers have changed from their
   original values i.e. *ANSCP or *ANSP2 has changed.

   If errors are encountered then *TERRNO is set to an appropriate
   errno value and a zero result is returned for a recoverable error,
   and a less-than zero result is returned for a non-recoverable error.

   If no errors are encountered then *TERRNO is left unmodified and
   a the length of the first response in bytes is returned.  */
static int
send_vc(res_state statp,
	const u_char *buf, int buflen, const u_char *buf2, int buflen2,
	u_char **ansp, int *anssizp,
	int *terrno, int ns, u_char **anscp, u_char **ansp2, int *anssizp2,
	int *resplen2, int *ansp2_malloced)
{
	const HEADER *hp = (HEADER *) buf;
	const HEADER *hp2 = (HEADER *) buf2;
	HEADER *anhp = (HEADER *) *ansp;
	struct sockaddr *nsap = __res_get_nsaddr (statp, ns);
	int truncating, connreset, n;
	/* On some architectures compiler might emit a warning indicating
	   'resplen' may be used uninitialized.  However if buf2 == NULL
	   then this code won't be executed; if buf2 != NULL, then first
	   time round the loop recvresp1 and recvresp2 will be 0 so this
	   code won't be executed but "thisresplenp = &resplen;" followed
	   by "*thisresplenp = rlen;" will be executed so that subsequent
	   times round the loop resplen has been initialized.  So this is
	   a false-positive.
	 */
	DIAG_PUSH_NEEDS_COMMENT;
#if defined(__clang__)
	DIAG_IGNORE_NEEDS_COMMENT (5, "-Wsometimes-uninitialized");
#else
	DIAG_IGNORE_NEEDS_COMMENT (5, "-Wmaybe-uninitialized");
#endif
	int resplen;
	DIAG_POP_NEEDS_COMMENT;
	struct iovec iov[4];
	u_short len;
	u_short len2;
	u_char *cp;

	connreset = 0;
 same_ns:
	truncating = 0;

	/* Are we still talking to whom we want to talk to? */
	if (statp->_vcsock >= 0 && (statp->_flags & RES_F_VC) != 0) {
		struct sockaddr_in6 peer;
		socklen_t size = sizeof peer;

		if (__getpeername (statp->_vcsock,
				   (struct sockaddr *) &peer, &size) < 0
		    || !sock_eq (&peer, (struct sockaddr_in6 *) nsap)) {
			__res_iclose(statp, false);
			statp->_flags &= ~RES_F_VC;
		}
	}

	if (statp->_vcsock < 0 || (statp->_flags & RES_F_VC) == 0) {
		if (statp->_vcsock >= 0)
		  __res_iclose(statp, false);

		statp->_vcsock = __socket
		  (nsap->sa_family, SOCK_STREAM | SOCK_CLOEXEC, 0);
		if (statp->_vcsock < 0) {
			*terrno = errno;
			if (resplen2 != NULL)
			  *resplen2 = 0;
			return (-1);
		}
		__set_errno (0);
		if (__connect (statp->_vcsock, nsap,
			       nsap->sa_family == AF_INET
			       ? sizeof (struct sockaddr_in)
			       : sizeof (struct sockaddr_in6)) < 0) {
			*terrno = errno;
			return close_and_return_error (statp, resplen2);
		}
		statp->_flags |= RES_F_VC;
	}

	/*
	 * Send length & message
	 */
	len = htons ((u_short) buflen);
	evConsIovec(&len, INT16SZ, &iov[0]);
	evConsIovec((void*)buf, buflen, &iov[1]);
	int niov = 2;
	ssize_t explen = INT16SZ + buflen;
	if (buf2 != NULL) {
		len2 = htons ((u_short) buflen2);
		evConsIovec(&len2, INT16SZ, &iov[2]);
		evConsIovec((void*)buf2, buflen2, &iov[3]);
		niov = 4;
		explen += INT16SZ + buflen2;
	}
	if (TEMP_FAILURE_RETRY (__writev (statp->_vcsock, iov, niov))
	    != explen) {
		*terrno = errno;
		return close_and_return_error (statp, resplen2);
	}
	/*
	 * Receive length & response
	 */
	int recvresp1 = 0;
	/* Skip the second response if there is no second query.
	   To do that we mark the second response as received.  */
	int recvresp2 = buf2 == NULL;
	uint16_t rlen16;
 read_len:
	cp = (u_char *)&rlen16;
	len = sizeof(rlen16);
	while ((n = TEMP_FAILURE_RETRY (read(statp->_vcsock, cp,
					     (int)len))) > 0) {
		cp += n;
		if ((len -= n) <= 0)
			break;
	}
	if (n <= 0) {
		*terrno = errno;
		/*
		 * A long running process might get its TCP
		 * connection reset if the remote server was
		 * restarted.  Requery the server instead of
		 * trying a new one.  When there is only one
		 * server, this means that a query might work
		 * instead of failing.  We only allow one reset
		 * per query to prevent looping.
		 */
		if (*terrno == ECONNRESET && !connreset)
		  {
		    __res_iclose (statp, false);
		    connreset = 1;
		    goto same_ns;
		  }
		return close_and_return_error (statp, resplen2);
	}
	int rlen = ntohs (rlen16);

	int *thisanssizp;
	u_char **thisansp;
	int *thisresplenp;
	if ((recvresp1 | recvresp2) == 0 || buf2 == NULL) {
		/* We have not received any responses
		   yet or we only have one response to
		   receive.  */
		thisanssizp = anssizp;
		thisansp = anscp ?: ansp;
		assert (anscp != NULL || ansp2 == NULL);
		thisresplenp = &resplen;
	} else {
		thisanssizp = anssizp2;
		thisansp = ansp2;
		thisresplenp = resplen2;
	}
	anhp = (HEADER *) *thisansp;

	*thisresplenp = rlen;
	/* Is the answer buffer too small?  */
	if (*thisanssizp < rlen) {
		/* If the current buffer is not the the static
		   user-supplied buffer then we can reallocate
		   it.  */
		if (thisansp != NULL && thisansp != ansp) {
			/* Always allocate MAXPACKET, callers expect
			   this specific size.  */
			u_char *newp = malloc (MAXPACKET);
			if (newp == NULL)
			  {
			    *terrno = ENOMEM;
			    return close_and_return_error (statp, resplen2);
			  }
			*thisanssizp = MAXPACKET;
			*thisansp = newp;
			if (thisansp == ansp2)
			  *ansp2_malloced = 1;
			anhp = (HEADER *) newp;
			/* A uint16_t can't be larger than MAXPACKET
			   thus it's safe to allocate MAXPACKET but
			   read RLEN bytes instead.  */
			len = rlen;
		} else {
			truncating = 1;
			len = *thisanssizp;
		}
	} else
		len = rlen;

	if (__glibc_unlikely (len < HFIXEDSZ))       {
		/*
		 * Undersized message.
		 */
		*terrno = EMSGSIZE;
		return close_and_return_error (statp, resplen2);
	}

	cp = *thisansp;
	while (len != 0 && (n = read(statp->_vcsock, (char *)cp, (int)len)) > 0){
		cp += n;
		len -= n;
	}
	if (__glibc_unlikely (n <= 0))       {
		*terrno = errno;
		return close_and_return_error (statp, resplen2);
	}
	if (__glibc_unlikely (truncating))       {
		/*
		 * Flush rest of answer so connection stays in synch.
		 */
		anhp->tc = 1;
		len = rlen - *thisanssizp;
		while (len != 0) {
			char junk[PACKETSZ];

			n = read(statp->_vcsock, junk,
				 (len > sizeof junk) ? sizeof junk : len);
			if (n > 0)
				len -= n;
			else
				break;
		}
	}
	/*
	 * If the calling application has bailed out of
	 * a previous call and failed to arrange to have
	 * the circuit closed or the server has got
	 * itself confused, then drop the packet and
	 * wait for the correct one.
	 */
	if ((recvresp1 || hp->id != anhp->id)
	    && (recvresp2 || hp2->id != anhp->id))
		goto read_len;

	/* Mark which reply we received.  */
	if (recvresp1 == 0 && hp->id == anhp->id)
	  recvresp1 = 1;
	else
	  recvresp2 = 1;
	/* Repeat waiting if we have a second answer to arrive.  */
	if ((recvresp1 & recvresp2) == 0)
		goto read_len;

	/*
	 * All is well, or the error is fatal.  Signal that the
	 * next nameserver ought not be tried.
	 */
	return resplen;
}

static int
reopen (res_state statp, int *terrno, int ns)
{
	if (EXT(statp).nssocks[ns] == -1) {
		struct sockaddr *nsap = __res_get_nsaddr (statp, ns);
		socklen_t slen;

		/* only try IPv6 if IPv6 NS and if not failed before */
		if (nsap->sa_family == AF_INET6 && !statp->ipv6_unavail) {
			EXT (statp).nssocks[ns] = __socket
			  (PF_INET6,
			   SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
			if (EXT(statp).nssocks[ns] < 0)
			    statp->ipv6_unavail = errno == EAFNOSUPPORT;
			slen = sizeof (struct sockaddr_in6);
		} else if (nsap->sa_family == AF_INET) {
			EXT (statp).nssocks[ns] = __socket
			  (PF_INET,
			   SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
			slen = sizeof (struct sockaddr_in);
		}
		if (EXT(statp).nssocks[ns] < 0) {
			*terrno = errno;
			return (-1);
		}

		/* Enable full ICMP error reporting for this
		   socket.  */
		if (__res_enable_icmp (nsap->sa_family,
				       EXT (statp).nssocks[ns]) < 0)
		  {
		    int saved_errno = errno;
		    __res_iclose (statp, false);
		    __set_errno (saved_errno);
		    *terrno = saved_errno;
		    return -1;
		  }

		/*
		 * On a 4.3BSD+ machine (client and server,
		 * actually), sending to a nameserver datagram
		 * port with no nameserver will cause an
		 * ICMP port unreachable message to be returned.
		 * If our datagram socket is "connected" to the
		 * server, we get an ECONNREFUSED error on the next
		 * socket operation, and select returns if the
		 * error message is received.  We can thus detect
		 * the absence of a nameserver without timing out.
		 */
		/* With GCC 5.3 when compiling with -Os the compiler
		   emits a warning that slen may be used uninitialized,
		   but that is never true.  Both slen and
		   EXT(statp).nssocks[ns] are initialized together or
		   the function return -1 before control flow reaches
		   the call to connect with slen.  */
		DIAG_PUSH_NEEDS_COMMENT;
		DIAG_IGNORE_Os_NEEDS_COMMENT (5, "-Wmaybe-uninitialized");
		if (__connect (EXT (statp).nssocks[ns], nsap, slen) < 0) {
		DIAG_POP_NEEDS_COMMENT;
			__res_iclose(statp, false);
			return (0);
		}
	}

	return 1;
}

/* The send_dg function is responsible for sending a DNS query over UDP
   to the nameserver numbered NS from the res_state STATP i.e.
   EXT(statp).nssocks[ns].  The function supports IPv4 and IPv6 queries
   along with the ability to send the query in parallel for both stacks
   (default) or serially (RES_SINGLKUP).  It also supports serial lookup
   with a close and reopen of the socket used to talk to the server
   (RES_SNGLKUPREOP) to work around broken name servers.

   The query stored in BUF of BUFLEN length is sent first followed by
   the query stored in BUF2 of BUFLEN2 length.  Queries are sent
   in parallel (default) or serially (RES_SINGLKUP or RES_SNGLKUPREOP).

   Answers to the query are stored firstly in *ANSP up to a max of
   *ANSSIZP bytes.  If more than *ANSSIZP bytes are needed and ANSCP
   is non-NULL (to indicate that modifying the answer buffer is allowed)
   then malloc is used to allocate a new response buffer and ANSCP and
   ANSP will both point to the new buffer.  If more than *ANSSIZP bytes
   are needed but ANSCP is NULL, then as much of the response as
   possible is read into the buffer, but the results will be truncated.
   When truncation happens because of a small answer buffer the DNS
   packets header field TC will bet set to 1, indicating a truncated
   message, while the rest of the UDP packet is discarded.

   Answers to the query are stored secondly in *ANSP2 up to a max of
   *ANSSIZP2 bytes, with the actual response length stored in
   *RESPLEN2.  If more than *ANSSIZP bytes are needed and ANSP2
   is non-NULL (required for a second query) then malloc is used to
   allocate a new response buffer, *ANSSIZP2 is set to the new buffer
   size and *ANSP2_MALLOCED is set to 1.

   The ANSP2_MALLOCED argument will eventually be removed as the
   change in buffer pointer can be used to detect the buffer has
   changed and that the caller should use free on the new buffer.

   Note that the answers may arrive in any order from the server and
   therefore the first and second answer buffers may not correspond to
   the first and second queries.

   It is not supported to call this function with a non-NULL ANSP2
   but a NULL ANSCP.  Put another way, you can call send_vc with a
   single unmodifiable buffer or two modifiable buffers, but no other
   combination is supported.

   It is the caller's responsibility to free the malloc allocated
   buffers by detecting that the pointers have changed from their
   original values i.e. *ANSCP or *ANSP2 has changed.

   If an answer is truncated because of UDP datagram DNS limits then
   *V_CIRCUIT is set to 1 and the return value non-zero to indicate to
   the caller to retry with TCP.  The value *GOTSOMEWHERE is set to 1
   if any progress was made reading a response from the nameserver and
   is used by the caller to distinguish between ECONNREFUSED and
   ETIMEDOUT (the latter if *GOTSOMEWHERE is 1).

   If errors are encountered then *TERRNO is set to an appropriate
   errno value and a zero result is returned for a recoverable error,
   and a less-than zero result is returned for a non-recoverable error.

   If no errors are encountered then *TERRNO is left unmodified and
   a the length of the first response in bytes is returned.  */
static int
send_dg(res_state statp,
	const u_char *buf, int buflen, const u_char *buf2, int buflen2,
	u_char **ansp, int *anssizp,
	int *terrno, int ns, int *v_circuit, int *gotsomewhere, u_char **anscp,
	u_char **ansp2, int *anssizp2, int *resplen2, int *ansp2_malloced)
{
	const HEADER *hp = (HEADER *) buf;
	const HEADER *hp2 = (HEADER *) buf2;
	struct timespec now, timeout, finish;
	struct pollfd pfd[1];
	int ptimeout;
	struct sockaddr_in6 from;
	int resplen = 0;
	int n;

	/*
	 * Compute time for the total operation.
	 */
	int seconds = (statp->retrans << ns);
	if (ns > 0)
		seconds /= statp->nscount;
	if (seconds <= 0)
		seconds = 1;
	bool single_request_reopen = (statp->options & RES_SNGLKUPREOP) != 0;
	bool single_request = (((statp->options & RES_SNGLKUP) != 0)
			       | single_request_reopen);
	int save_gotsomewhere = *gotsomewhere;

	int retval;
 retry_reopen:
	retval = reopen (statp, terrno, ns);
	if (retval <= 0)
	  {
	    if (resplen2 != NULL)
	      *resplen2 = 0;
	    return retval;
	  }
 retry:
	evNowTime(&now);
	evConsTime(&timeout, seconds, 0);
	evAddTime(&finish, &now, &timeout);
	int need_recompute = 0;
	int nwritten = 0;
	int recvresp1 = 0;
	/* Skip the second response if there is no second query.
	   To do that we mark the second response as received.  */
	int recvresp2 = buf2 == NULL;
	pfd[0].fd = EXT(statp).nssocks[ns];
	pfd[0].events = POLLOUT;
 wait:
	if (need_recompute) {
	recompute_resend:
		evNowTime(&now);
		if (evCmpTime(finish, now) <= 0) {
		poll_err_out:
			return close_and_return_error (statp, resplen2);
		}
		evSubTime(&timeout, &finish, &now);
		need_recompute = 0;
	}
	/* Convert struct timespec in milliseconds.  */
	ptimeout = timeout.tv_sec * 1000 + timeout.tv_nsec / 1000000;

	n = 0;
	if (nwritten == 0)
	  n = __poll (pfd, 1, 0);
	if (__glibc_unlikely (n == 0))       {
		n = __poll (pfd, 1, ptimeout);
		need_recompute = 1;
	}
	if (n == 0) {
		if (resplen > 1 && (recvresp1 || (buf2 != NULL && recvresp2)))
		  {
		    /* There are quite a few broken name servers out
		       there which don't handle two outstanding
		       requests from the same source.  There are also
		       broken firewall settings.  If we time out after
		       having received one answer switch to the mode
		       where we send the second request only once we
		       have received the first answer.  */
		    if (!single_request)
		      {
			statp->options |= RES_SNGLKUP;
			single_request = true;
			*gotsomewhere = save_gotsomewhere;
			goto retry;
		      }
		    else if (!single_request_reopen)
		      {
			statp->options |= RES_SNGLKUPREOP;
			single_request_reopen = true;
			*gotsomewhere = save_gotsomewhere;
			__res_iclose (statp, false);
			goto retry_reopen;
		      }

		    *resplen2 = 1;
		    return resplen;
		  }

		*gotsomewhere = 1;
		if (resplen2 != NULL)
		  *resplen2 = 0;
		return 0;
	}
	if (n < 0) {
		if (errno == EINTR)
			goto recompute_resend;

		goto poll_err_out;
	}
	__set_errno (0);
	if (pfd[0].revents & POLLOUT) {
#ifndef __ASSUME_SENDMMSG
		static int have_sendmmsg;
#else
# define have_sendmmsg 1
#endif
		if (have_sendmmsg >= 0 && nwritten == 0 && buf2 != NULL
		    && !single_request)
		  {
		    struct iovec iov =
		      { .iov_base = (void *) buf, .iov_len = buflen };
		    struct iovec iov2 =
		      { .iov_base = (void *) buf2, .iov_len = buflen2 };
		    struct mmsghdr reqs[2] =
		      {
			{
			  .msg_hdr =
			    {
			      .msg_iov = &iov,
			      .msg_iovlen = 1,
			    },
			},
			{
			  .msg_hdr =
			    {
			      .msg_iov = &iov2,
			      .msg_iovlen = 1,
			    }
			},
		      };

		    int ndg = __sendmmsg (pfd[0].fd, reqs, 2, MSG_NOSIGNAL);
		    if (__glibc_likely (ndg == 2))
		      {
			if (reqs[0].msg_len != buflen
			    || reqs[1].msg_len != buflen2)
			  goto fail_sendmmsg;

			pfd[0].events = POLLIN;
			nwritten += 2;
		      }
		    else if (ndg == 1 && reqs[0].msg_len == buflen)
		      goto just_one;
		    else if (ndg < 0 && (errno == EINTR || errno == EAGAIN))
		      goto recompute_resend;
		    else
		      {
#ifndef __ASSUME_SENDMMSG
			if (__glibc_unlikely (have_sendmmsg == 0))
			  {
			    if (ndg < 0 && errno == ENOSYS)
			      {
				have_sendmmsg = -1;
				goto try_send;
			      }
			    have_sendmmsg = 1;
			  }
#endif

		      fail_sendmmsg:
			return close_and_return_error (statp, resplen2);
		      }
		  }
		else
		  {
		    ssize_t sr;
#ifndef __ASSUME_SENDMMSG
		  try_send:
#endif
		    if (nwritten != 0)
		      sr = __send (pfd[0].fd, buf2, buflen2, MSG_NOSIGNAL);
		    else
		      sr = __send (pfd[0].fd, buf, buflen, MSG_NOSIGNAL);

		    if (sr != (nwritten != 0 ? buflen2 : buflen)) {
		      if (errno == EINTR || errno == EAGAIN)
			goto recompute_resend;
		      return close_and_return_error (statp, resplen2);
		    }
		  just_one:
		    if (nwritten != 0 || buf2 == NULL || single_request)
		      pfd[0].events = POLLIN;
		    else
		      pfd[0].events = POLLIN | POLLOUT;
		    ++nwritten;
		  }
		goto wait;
	} else if (pfd[0].revents & POLLIN) {
		int *thisanssizp;
		u_char **thisansp;
		int *thisresplenp;

		if ((recvresp1 | recvresp2) == 0 || buf2 == NULL) {
			/* We have not received any responses
			   yet or we only have one response to
			   receive.  */
			thisanssizp = anssizp;
			thisansp = anscp ?: ansp;
			assert (anscp != NULL || ansp2 == NULL);
			thisresplenp = &resplen;
		} else {
			thisanssizp = anssizp2;
			thisansp = ansp2;
			thisresplenp = resplen2;
		}

		if (*thisanssizp < MAXPACKET
		    /* If the current buffer is not the the static
		       user-supplied buffer then we can reallocate
		       it.  */
		    && (thisansp != NULL && thisansp != ansp)
#ifdef FIONREAD
		    /* Is the size too small?  */
		    && (__ioctl (pfd[0].fd, FIONREAD, thisresplenp) < 0
			|| *thisanssizp < *thisresplenp)
#endif
                    ) {
			/* Always allocate MAXPACKET, callers expect
			   this specific size.  */
			u_char *newp = malloc (MAXPACKET);
			if (newp != NULL) {
				*thisanssizp = MAXPACKET;
				*thisansp = newp;
				if (thisansp == ansp2)
				  *ansp2_malloced = 1;
			}
		}
		/* We could end up with truncation if anscp was NULL
		   (not allowed to change caller's buffer) and the
		   response buffer size is too small.  This isn't a
		   reliable way to detect truncation because the ioctl
		   may be an inaccurate report of the UDP message size.
		   Therefore we use this only to issue debug output.
		   To do truncation accurately with UDP we need
		   MSG_TRUNC which is only available on Linux.  We
		   can abstract out the Linux-specific feature in the
		   future to detect truncation.  */
		HEADER *anhp = (HEADER *) *thisansp;
		socklen_t fromlen = sizeof(struct sockaddr_in6);
		assert (sizeof(from) <= fromlen);
		*thisresplenp = __recvfrom (pfd[0].fd, (char *) *thisansp,
					    *thisanssizp, 0,
					    (struct sockaddr *) &from,
					    &fromlen);
		if (__glibc_unlikely (*thisresplenp <= 0))       {
			if (errno == EINTR || errno == EAGAIN) {
				need_recompute = 1;
				goto wait;
			}
			return close_and_return_error (statp, resplen2);
		}
		*gotsomewhere = 1;
		if (__glibc_unlikely (*thisresplenp < HFIXEDSZ))       {
			/*
			 * Undersized message.
			 */
			*terrno = EMSGSIZE;
			return close_and_return_error (statp, resplen2);
		}

		/* Check for the correct header layout and a matching
		   question.  */
		int matching_query = 0; /* Default to no matching query.  */
		if (!recvresp1
		    && anhp->id == hp->id
		    && __libc_res_queriesmatch (buf, buf + buflen,
						*thisansp,
						*thisansp + *thisanssizp))
		  matching_query = 1;
		if (!recvresp2
		    && anhp->id == hp2->id
		    && __libc_res_queriesmatch (buf2, buf2 + buflen2,
						*thisansp,
						*thisansp + *thisanssizp))
		  matching_query = 2;
		if (matching_query == 0)
		  /* Spurious UDP packet.  Drop it and continue
		     waiting.  */
		  {
		    need_recompute = 1;
		    goto wait;
		  }

		if (anhp->rcode == SERVFAIL ||
		    anhp->rcode == NOTIMP ||
		    anhp->rcode == REFUSED) {
		next_ns:
			if (recvresp1 || (buf2 != NULL && recvresp2)) {
			  *resplen2 = 0;
			  return resplen;
			}
			if (buf2 != NULL)
			  {
			    /* No data from the first reply.  */
			    resplen = 0;
			    /* We are waiting for a possible second reply.  */
			    if (matching_query == 1)
			      recvresp1 = 1;
			    else
			      recvresp2 = 1;

			    goto wait;
			  }

			/* don't retry if called from dig */
			if (!statp->pfcode)
			  return close_and_return_error (statp, resplen2);
			__res_iclose(statp, false);
		}
		if (anhp->rcode == NOERROR && anhp->ancount == 0
		    && anhp->aa == 0 && anhp->ra == 0 && anhp->arcount == 0) {
			goto next_ns;
		}
		if (!(statp->options & RES_IGNTC) && anhp->tc) {
			/*
			 * To get the rest of answer,
			 * use TCP with same server.
			 */
			*v_circuit = 1;
			__res_iclose(statp, false);
			// XXX if we have received one reply we could
			// XXX use it and not repeat it over TCP...
			if (resplen2 != NULL)
			  *resplen2 = 0;
			return (1);
		}
		/* Mark which reply we received.  */
		if (matching_query == 1)
			recvresp1 = 1;
		else
			recvresp2 = 1;
		/* Repeat waiting if we have a second answer to arrive.  */
		if ((recvresp1 & recvresp2) == 0) {
			if (single_request) {
				pfd[0].events = POLLOUT;
				if (single_request_reopen) {
					__res_iclose (statp, false);
					retval = reopen (statp, terrno, ns);
					if (retval <= 0)
					  {
					    if (resplen2 != NULL)
					      *resplen2 = 0;
					    return retval;
					  }
					pfd[0].fd = EXT(statp).nssocks[ns];
				}
			}
			goto wait;
		}
		/* All is well.  We have received both responses (if
		   two responses were requested).  */
		return (resplen);
	} else if (pfd[0].revents & (POLLERR | POLLHUP | POLLNVAL))
	  /* Something went wrong.  We can stop trying.  */
	  return close_and_return_error (statp, resplen2);
	else {
		/* poll should not have returned > 0 in this case.  */
		abort ();
	}
}

static int
sock_eq(struct sockaddr_in6 *a1, struct sockaddr_in6 *a2) {
	if (a1->sin6_family == a2->sin6_family) {
		if (a1->sin6_family == AF_INET)
			return ((((struct sockaddr_in *)a1)->sin_port ==
				 ((struct sockaddr_in *)a2)->sin_port) &&
				(((struct sockaddr_in *)a1)->sin_addr.s_addr ==
				 ((struct sockaddr_in *)a2)->sin_addr.s_addr));
		else
			return ((a1->sin6_port == a2->sin6_port) &&
				!memcmp(&a1->sin6_addr, &a2->sin6_addr,
					sizeof (struct in6_addr)));
	}
	if (a1->sin6_family == AF_INET) {
		struct sockaddr_in6 *sap = a1;
		a1 = a2;
		a2 = sap;
	} /* assumes that AF_INET and AF_INET6 are the only possibilities */
	return ((a1->sin6_port == ((struct sockaddr_in *)a2)->sin_port) &&
		IN6_IS_ADDR_V4MAPPED(&a1->sin6_addr) &&
		(a1->sin6_addr.s6_addr32[3] ==
		 ((struct sockaddr_in *)a2)->sin_addr.s_addr));
}
