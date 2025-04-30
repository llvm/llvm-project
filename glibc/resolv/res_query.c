/*
 * Copyright (c) 1988, 1993
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

#include <assert.h>
#include <sys/types.h>
#include <sys/param.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <arpa/nameser.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <resolv.h>
#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <shlib-compat.h>

#if PACKETSZ > 65536
#define MAXPACKET	PACKETSZ
#else
#define MAXPACKET	65536
#endif

#define QUERYSIZE	(HFIXEDSZ + QFIXEDSZ + MAXCDNAME + 1)

static int
__res_context_querydomain (struct resolv_context *,
			   const char *name, const char *domain,
			   int class, int type, unsigned char *answer, int anslen,
			   unsigned char **answerp, unsigned char **answerp2, int *nanswerp2,
			   int *resplen2, int *answerp2_malloced);

/* Formulate a normal query, send, and await answer.  Returned answer
   is placed in supplied buffer ANSWER.  Perform preliminary check of
   answer, returning success only if no error is indicated and the
   answer count is nonzero.  Return the size of the response on
   success, -1 on error.  Error number is left in h_errno.

   Caller must parse answer and determine whether it answers the
   question.  */
int
__res_context_query (struct resolv_context *ctx, const char *name,
		     int class, int type,
		     unsigned char *answer, int anslen,
		     unsigned char **answerp, unsigned char **answerp2,
		     int *nanswerp2, int *resplen2, int *answerp2_malloced)
{
	struct __res_state *statp = ctx->resp;
	HEADER *hp = (HEADER *) answer;
	HEADER *hp2;
	int n, use_malloc = 0;

	size_t bufsize = (type == T_QUERY_A_AND_AAAA ? 2 : 1) * QUERYSIZE;
	u_char *buf = alloca (bufsize);
	u_char *query1 = buf;
	int nquery1 = -1;
	u_char *query2 = NULL;
	int nquery2 = 0;

 again:
	hp->rcode = NOERROR;	/* default */

	if (type == T_QUERY_A_AND_AAAA)
	  {
	    n = __res_context_mkquery (ctx, QUERY, name, class, T_A, NULL,
				       query1, bufsize);
	    if (n > 0)
	      {
		if ((statp->options & (RES_USE_EDNS0|RES_USE_DNSSEC)) != 0)
		  {
		    /* Use RESOLV_EDNS_BUFFER_SIZE because the receive
		       buffer can be reallocated.  */
		    n = __res_nopt (ctx, n, query1, bufsize,
				    RESOLV_EDNS_BUFFER_SIZE);
		    if (n < 0)
		      goto unspec_nomem;
		  }

		nquery1 = n;
		/* Align the buffer.  */
		int npad = ((nquery1 + __alignof__ (HEADER) - 1)
			    & ~(__alignof__ (HEADER) - 1)) - nquery1;
		if (n > bufsize - npad)
		  {
		    n = -1;
		    goto unspec_nomem;
		  }
		int nused = n + npad;
		query2 = buf + nused;
		n = __res_context_mkquery (ctx, QUERY, name, class, T_AAAA,
					   NULL, query2, bufsize - nused);
		if (n > 0
		    && (statp->options & (RES_USE_EDNS0|RES_USE_DNSSEC)) != 0)
		  /* Use RESOLV_EDNS_BUFFER_SIZE because the receive
		     buffer can be reallocated.  */
		  n = __res_nopt (ctx, n, query2, bufsize,
				  RESOLV_EDNS_BUFFER_SIZE);
		nquery2 = n;
	      }

	  unspec_nomem:;
	  }
	else
	  {
	    n = __res_context_mkquery (ctx, QUERY, name, class, type, NULL,
				       query1, bufsize);

	    if (n > 0
		&& (statp->options & (RES_USE_EDNS0|RES_USE_DNSSEC)) != 0)
	      {
		/* Use RESOLV_EDNS_BUFFER_SIZE if the receive buffer
		   can be reallocated.  */
		size_t advertise;
		if (answerp == NULL)
		  advertise = anslen;
		else
		  advertise = RESOLV_EDNS_BUFFER_SIZE;
		n = __res_nopt (ctx, n, query1, bufsize, advertise);
	      }

	    nquery1 = n;
	  }

	if (__glibc_unlikely (n <= 0) && !use_malloc) {
		/* Retry just in case res_nmkquery failed because of too
		   short buffer.  Shouldn't happen.  */
		bufsize = (type == T_QUERY_A_AND_AAAA ? 2 : 1) * MAXPACKET;
		buf = malloc (bufsize);
		if (buf != NULL) {
			query1 = buf;
			use_malloc = 1;
			goto again;
		}
	}
	if (__glibc_unlikely (n <= 0))       {
		RES_SET_H_ERRNO(statp, NO_RECOVERY);
		if (use_malloc)
			free (buf);
		return (n);
	}
	assert (answerp == NULL || (void *) *answerp == (void *) answer);
	n = __res_context_send (ctx, query1, nquery1, query2, nquery2, answer,
				anslen, answerp, answerp2, nanswerp2, resplen2,
				answerp2_malloced);
	if (use_malloc)
		free (buf);
	if (n < 0) {
		RES_SET_H_ERRNO(statp, TRY_AGAIN);
		return (n);
	}

	if (answerp != NULL)
	  /* __res_context_send might have reallocated the buffer.  */
	  hp = (HEADER *) *answerp;

	/* We simplify the following tests by assigning HP to HP2 or
	   vice versa.  It is easy to verify that this is the same as
	   ignoring all tests of HP or HP2.  */
	if (answerp2 == NULL || *resplen2 < (int) sizeof (HEADER))
	  {
	    hp2 = hp;
	  }
	else
	  {
	    hp2 = (HEADER *) *answerp2;
	    if (n < (int) sizeof (HEADER))
	      {
	        hp = hp2;
	      }
	  }

	/* Make sure both hp and hp2 are defined */
	assert((hp != NULL) && (hp2 != NULL));

	if ((hp->rcode != NOERROR || ntohs(hp->ancount) == 0)
	    && (hp2->rcode != NOERROR || ntohs(hp2->ancount) == 0)) {
		switch (hp->rcode == NOERROR ? hp2->rcode : hp->rcode) {
		case NXDOMAIN:
			if ((hp->rcode == NOERROR && ntohs (hp->ancount) != 0)
			    || (hp2->rcode == NOERROR
				&& ntohs (hp2->ancount) != 0))
				goto success;
			RES_SET_H_ERRNO(statp, HOST_NOT_FOUND);
			break;
		case SERVFAIL:
			RES_SET_H_ERRNO(statp, TRY_AGAIN);
			break;
		case NOERROR:
			if (ntohs (hp->ancount) != 0
			    || ntohs (hp2->ancount) != 0)
				goto success;
			RES_SET_H_ERRNO(statp, NO_DATA);
			break;
		case FORMERR:
		case NOTIMP:
			/* Servers must not reply to AAAA queries with
			   NOTIMP etc but some of them do.  */
			if ((hp->rcode == NOERROR && ntohs (hp->ancount) != 0)
			    || (hp2->rcode == NOERROR
				&& ntohs (hp2->ancount) != 0))
				goto success;
			/* FALLTHROUGH */
		case REFUSED:
		default:
			RES_SET_H_ERRNO(statp, NO_RECOVERY);
			break;
		}
		return (-1);
	}
 success:
	return (n);
}
libc_hidden_def (__res_context_query)

/* Common part of res_nquery and res_query.  */
static int
context_query_common (struct resolv_context *ctx,
		      const char *name, int class, int type,
		      unsigned char *answer, int anslen)
{
  if (ctx == NULL)
    {
      RES_SET_H_ERRNO (&_res, NETDB_INTERNAL);
      return -1;
    }
  int result = __res_context_query (ctx, name, class, type, answer, anslen,
				    NULL, NULL, NULL, NULL, NULL);
  __resolv_context_put (ctx);
  return result;
}

int
___res_nquery (res_state statp,
	       const char *name,      /* Domain name.  */
	       int class, int type,   /* Class and type of query.  */
	       unsigned char *answer, /* Buffer to put answer.  */
	       int anslen)	      /* Size of answer buffer.  */
{
  return context_query_common
    (__resolv_context_get_override (statp), name, class, type, answer, anslen);
}
versioned_symbol (libc, ___res_nquery, res_nquery, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_nquery, __res_nquery, GLIBC_2_2);
#endif

int
___res_query (const char *name, int class, int type,
	      unsigned char *answer, int anslen)
{
  return context_query_common
    (__resolv_context_get (), name, class, type, answer, anslen);
}
versioned_symbol (libc, ___res_query, res_query, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_2)
compat_symbol (libresolv, ___res_query, res_query, GLIBC_2_0);
#endif
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_query, __res_query, GLIBC_2_2);
#endif

/* Formulate a normal query, send, and retrieve answer in supplied
   buffer.  Return the size of the response on success, -1 on error.
   If enabled, implement search rules until answer or unrecoverable
   failure is detected.  Error code, if any, is left in h_errno.  */
int
__res_context_search (struct resolv_context *ctx,
		      const char *name, int class, int type,
		      unsigned char *answer, int anslen,
		      unsigned char **answerp, unsigned char **answerp2,
		      int *nanswerp2, int *resplen2, int *answerp2_malloced)
{
	struct __res_state *statp = ctx->resp;
	const char *cp;
	HEADER *hp = (HEADER *) answer;
	char tmp[NS_MAXDNAME];
	u_int dots;
	int trailing_dot, ret, saved_herrno;
	int got_nodata = 0, got_servfail = 0, root_on_list = 0;
	int tried_as_is = 0;
	int searched = 0;

	__set_errno (0);
	RES_SET_H_ERRNO(statp, HOST_NOT_FOUND);  /* True if we never query. */

	dots = 0;
	for (cp = name; *cp != '\0'; cp++)
		dots += (*cp == '.');
	trailing_dot = 0;
	if (cp > name && *--cp == '.')
		trailing_dot++;

	/* If there aren't any dots, it could be a user-level alias. */
	if (!dots && (cp = __res_context_hostalias
		      (ctx, name, tmp, sizeof tmp))!= NULL)
	  return __res_context_query (ctx, cp, class, type, answer,
				      anslen, answerp, answerp2,
				      nanswerp2, resplen2, answerp2_malloced);

	/*
	 * If there are enough dots in the name, let's just give it a
	 * try 'as is'. The threshold can be set with the "ndots" option.
	 * Also, query 'as is', if there is a trailing dot in the name.
	 */
	saved_herrno = -1;
	if (dots >= statp->ndots || trailing_dot) {
		ret = __res_context_querydomain (ctx, name, NULL, class, type,
						 answer, anslen, answerp,
						 answerp2, nanswerp2, resplen2,
						 answerp2_malloced);
		if (ret > 0 || trailing_dot
		    /* If the second response is valid then we use that.  */
		    || (ret == 0 && resplen2 != NULL && *resplen2 > 0))
			return (ret);
		saved_herrno = h_errno;
		tried_as_is++;
		if (answerp && *answerp != answer) {
			answer = *answerp;
			anslen = MAXPACKET;
		}
		if (answerp2 && *answerp2_malloced)
		  {
		    free (*answerp2);
		    *answerp2 = NULL;
		    *nanswerp2 = 0;
		    *answerp2_malloced = 0;
		  }
	}

	/*
	 * We do at least one level of search if
	 *	- there is no dot and RES_DEFNAME is set, or
	 *	- there is at least one dot, there is no trailing dot,
	 *	  and RES_DNSRCH is set.
	 */
	if ((!dots && (statp->options & RES_DEFNAMES) != 0) ||
	    (dots && !trailing_dot && (statp->options & RES_DNSRCH) != 0)) {
		int done = 0;

		for (size_t domain_index = 0; !done; ++domain_index) {
			const char *dname = __resolv_context_search_list
			  (ctx, domain_index);
			if (dname == NULL)
			  break;
			searched = 1;

			/* __res_context_querydoman concatenates name
			   with dname with a "." in between.  If we
			   pass it in dname the "." we got from the
			   configured default search path, we'll end
			   up with "name..", which won't resolve.
			   OTOH, passing it "" will result in "name.",
			   which has the intended effect for both
			   possible representations of the root
			   domain.  */
			if (dname[0] == '.')
				dname++;
			if (dname[0] == '\0')
				root_on_list++;

			ret = __res_context_querydomain
			  (ctx, name, dname, class, type,
			   answer, anslen, answerp, answerp2, nanswerp2,
			   resplen2, answerp2_malloced);
			if (ret > 0 || (ret == 0 && resplen2 != NULL
					&& *resplen2 > 0))
				return (ret);

			if (answerp && *answerp != answer) {
				answer = *answerp;
				anslen = MAXPACKET;
			}
			if (answerp2 && *answerp2_malloced)
			  {
			    free (*answerp2);
			    *answerp2 = NULL;
			    *nanswerp2 = 0;
			    *answerp2_malloced = 0;
			  }

			/*
			 * If no server present, give up.
			 * If name isn't found in this domain,
			 * keep trying higher domains in the search list
			 * (if that's enabled).
			 * On a NO_DATA error, keep trying, otherwise
			 * a wildcard entry of another type could keep us
			 * from finding this entry higher in the domain.
			 * If we get some other error (negative answer or
			 * server failure), then stop searching up,
			 * but try the input name below in case it's
			 * fully-qualified.
			 */
			if (errno == ECONNREFUSED) {
				RES_SET_H_ERRNO(statp, TRY_AGAIN);
				return (-1);
			}

			switch (statp->res_h_errno) {
			case NO_DATA:
				got_nodata++;
				/* FALLTHROUGH */
			case HOST_NOT_FOUND:
				/* keep trying */
				break;
			case TRY_AGAIN:
				if (hp->rcode == SERVFAIL) {
					/* try next search element, if any */
					got_servfail++;
					break;
				}
				/* FALLTHROUGH */
			default:
				/* anything else implies that we're done */
				done++;
			}

			/* if we got here for some reason other than DNSRCH,
			 * we only wanted one iteration of the loop, so stop.
			 */
			if ((statp->options & RES_DNSRCH) == 0)
				done++;
		}
	}

	/*
	 * If the query has not already been tried as is then try it
	 * unless RES_NOTLDQUERY is set and there were no dots.
	 */
	if ((dots || !searched || (statp->options & RES_NOTLDQUERY) == 0)
	    && !(tried_as_is || root_on_list)) {
		ret = __res_context_querydomain
		  (ctx, name, NULL, class, type,
		   answer, anslen, answerp, answerp2, nanswerp2,
		   resplen2, answerp2_malloced);
		if (ret > 0 || (ret == 0 && resplen2 != NULL
				&& *resplen2 > 0))
			return (ret);
	}

	/* if we got here, we didn't satisfy the search.
	 * if we did an initial full query, return that query's H_ERRNO
	 * (note that we wouldn't be here if that query had succeeded).
	 * else if we ever got a nodata, send that back as the reason.
	 * else send back meaningless H_ERRNO, that being the one from
	 * the last DNSRCH we did.
	 */
	if (answerp2 && *answerp2_malloced)
	  {
	    free (*answerp2);
	    *answerp2 = NULL;
	    *nanswerp2 = 0;
	    *answerp2_malloced = 0;
	  }
	if (saved_herrno != -1)
		RES_SET_H_ERRNO(statp, saved_herrno);
	else if (got_nodata)
		RES_SET_H_ERRNO(statp, NO_DATA);
	else if (got_servfail)
		RES_SET_H_ERRNO(statp, TRY_AGAIN);
	return (-1);
}
libc_hidden_def (__res_context_search)

/* Common part of res_nsearch and res_search.  */
static int
context_search_common (struct resolv_context *ctx,
		       const char *name, int class, int type,
		       unsigned char *answer, int anslen)
{
  if (ctx == NULL)
    {
      RES_SET_H_ERRNO (&_res, NETDB_INTERNAL);
      return -1;
    }
  int result = __res_context_search (ctx, name, class, type, answer, anslen,
				     NULL, NULL, NULL, NULL, NULL);
  __resolv_context_put (ctx);
  return result;
}

int
___res_nsearch (res_state statp,
		const char *name,      /* Domain name.  */
		int class, int type,   /* Class and type of query.  */
		unsigned char *answer, /* Buffer to put answer.  */
		int anslen)	       /* Size of answer.  */
{
  return context_search_common
    (__resolv_context_get_override (statp), name, class, type, answer, anslen);
}
versioned_symbol (libc, ___res_nsearch, res_nsearch, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_nsearch, __res_nsearch, GLIBC_2_2);
#endif

int
___res_search (const char *name, int class, int type,
	       unsigned char *answer, int anslen)
{
  return context_search_common
    (__resolv_context_get (), name, class, type, answer, anslen);
}
versioned_symbol (libc, ___res_search, res_search, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_2)
compat_symbol (libresolv, ___res_search, res_search, GLIBC_2_0);
#endif
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_search, __res_search, GLIBC_2_2);
#endif

/*  Perform a call on res_query on the concatenation of name and
    domain.  */
static int
__res_context_querydomain (struct resolv_context *ctx,
			   const char *name, const char *domain,
			   int class, int type,
			   unsigned char *answer, int anslen,
			   unsigned char **answerp, unsigned char **answerp2,
			   int *nanswerp2, int *resplen2,
			   int *answerp2_malloced)
{
	struct __res_state *statp = ctx->resp;
	char nbuf[MAXDNAME];
	const char *longname = nbuf;
	size_t n, d;

	if (domain == NULL) {
		n = strlen(name);

		/* Decrement N prior to checking it against MAXDNAME
		   so that we detect a wrap to SIZE_MAX and return
		   a reasonable error.  */
		n--;
		if (n >= MAXDNAME - 1) {
			RES_SET_H_ERRNO(statp, NO_RECOVERY);
			return (-1);
		}
		longname = name;
	} else {
		n = strlen(name);
		d = strlen(domain);
		if (n + d + 1 >= MAXDNAME) {
			RES_SET_H_ERRNO(statp, NO_RECOVERY);
			return (-1);
		}
		sprintf(nbuf, "%s.%s", name, domain);
	}
	return __res_context_query (ctx, longname, class, type, answer,
				    anslen, answerp, answerp2, nanswerp2,
				    resplen2, answerp2_malloced);
}

/* Common part of res_nquerydomain and res_querydomain.  */
static int
context_querydomain_common (struct resolv_context *ctx,
			    const char *name, const char *domain,
			    int class, int type,
			    unsigned char *answer, int anslen)
{
  if (ctx == NULL)
    {
      RES_SET_H_ERRNO (&_res, NETDB_INTERNAL);
      return -1;
    }
  int result = __res_context_querydomain (ctx, name, domain, class, type,
					  answer, anslen,
					  NULL, NULL, NULL, NULL, NULL);
  __resolv_context_put (ctx);
  return result;
}

int
___res_nquerydomain (res_state statp,
		     const char *name,
		     const char *domain,
		     int class, int type, /* Class and type of query.  */
		     unsigned char *answer, /* Buffer to put answer.  */
		     int anslen)	    /* Size of answer.  */
{
  return context_querydomain_common
    (__resolv_context_get_override (statp),
     name, domain, class, type, answer, anslen);
}
versioned_symbol (libc, ___res_nquerydomain, res_nquerydomain, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_nquerydomain, __res_nquerydomain, GLIBC_2_2);
#endif

int
___res_querydomain (const char *name, const char *domain, int class, int type,
		    unsigned char *answer, int anslen)
{
  return context_querydomain_common
    (__resolv_context_get (), name, domain, class, type, answer, anslen);
}
versioned_symbol (libc, ___res_querydomain, res_querydomain, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_2)
compat_symbol (libresolv, ___res_querydomain, res_querydomain, GLIBC_2_0);
#endif
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_querydomain, __res_querydomain, GLIBC_2_2);
#endif
