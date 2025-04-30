/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Extended from original form by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

/* Parts of this file are plain copies of the file `gethtnamadr.c' from
   the bind package and it has the following copyright.  */

/*
 * ++Copyright++ 1985, 1988, 1993
 * -
 * Copyright (c) 1985, 1988, 1993
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
 * -
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
 * -
 * --Copyright--
 */

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <libc-pointer-arith.h>

#include "nsswitch.h"
#include <arpa/nameser.h>
#include <nss_dns.h>

#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>

/* Get implementations of some internal functions.  */
#include <resolv/mapv4v6addr.h>
#include <resolv/mapv4v6hostent.h>

#define RESOLVSORT

#if PACKETSZ > 65536
# define MAXPACKET	PACKETSZ
#else
# define MAXPACKET	65536
#endif
/* As per RFC 1034 and 1035 a host name cannot exceed 255 octets in length.  */
#ifdef MAXHOSTNAMELEN
# undef MAXHOSTNAMELEN
#endif
#define MAXHOSTNAMELEN 256

/* We need this time later.  */
typedef union querybuf
{
  HEADER hdr;
  u_char buf[MAXPACKET];
} querybuf;

static enum nss_status getanswer_r (struct resolv_context *ctx,
				    const querybuf *answer, int anslen,
				    const char *qname, int qtype,
				    struct hostent *result, char *buffer,
				    size_t buflen, int *errnop, int *h_errnop,
				    int map, int32_t *ttlp, char **canonp);

static enum nss_status gaih_getanswer (const querybuf *answer1, int anslen1,
				       const querybuf *answer2, int anslen2,
				       const char *qname,
				       struct gaih_addrtuple **pat,
				       char *buffer, size_t buflen,
				       int *errnop, int *h_errnop,
				       int32_t *ttlp);

static enum nss_status gethostbyname3_context (struct resolv_context *ctx,
					       const char *name, int af,
					       struct hostent *result,
					       char *buffer, size_t buflen,
					       int *errnop, int *h_errnop,
					       int32_t *ttlp,
					       char **canonp);

/* Return the expected RDATA length for an address record type (A or
   AAAA).  */
static int
rrtype_to_rdata_length (int type)
{
  switch (type)
    {
    case T_A:
      return INADDRSZ;
    case T_AAAA:
      return IN6ADDRSZ;
    default:
      return -1;
    }
}


enum nss_status
_nss_dns_gethostbyname3_r (const char *name, int af, struct hostent *result,
			   char *buffer, size_t buflen, int *errnop,
			   int *h_errnop, int32_t *ttlp, char **canonp)
{
  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      *errnop = errno;
      *h_errnop = NETDB_INTERNAL;
      return NSS_STATUS_UNAVAIL;
    }
  enum nss_status status = gethostbyname3_context
    (ctx, name, af, result, buffer, buflen, errnop, h_errnop, ttlp, canonp);
  __resolv_context_put (ctx);
  return status;
}
libc_hidden_def (_nss_dns_gethostbyname3_r)

static enum nss_status
gethostbyname3_context (struct resolv_context *ctx,
			const char *name, int af, struct hostent *result,
			char *buffer, size_t buflen, int *errnop,
			int *h_errnop, int32_t *ttlp, char **canonp)
{
  union
  {
    querybuf *buf;
    u_char *ptr;
  } host_buffer;
  querybuf *orig_host_buffer;
  char tmp[NS_MAXDNAME];
  int size, type, n;
  const char *cp;
  int map = 0;
  int olderr = errno;
  enum nss_status status;

  switch (af) {
  case AF_INET:
    size = INADDRSZ;
    type = T_A;
    break;
  case AF_INET6:
    size = IN6ADDRSZ;
    type = T_AAAA;
    break;
  default:
    *h_errnop = NO_DATA;
    *errnop = EAFNOSUPPORT;
    return NSS_STATUS_UNAVAIL;
  }

  result->h_addrtype = af;
  result->h_length = size;

  /*
   * if there aren't any dots, it could be a user-level alias.
   * this is also done in res_query() since we are not the only
   * function that looks up host names.
   */
  if (strchr (name, '.') == NULL
      && (cp = __res_context_hostalias (ctx, name, tmp, sizeof (tmp))) != NULL)
    name = cp;

  host_buffer.buf = orig_host_buffer = (querybuf *) alloca (1024);

  n = __res_context_search (ctx, name, C_IN, type, host_buffer.buf->buf,
			    1024, &host_buffer.ptr, NULL, NULL, NULL, NULL);
  if (n < 0)
    {
      switch (errno)
	{
	case ESRCH:
	  status = NSS_STATUS_TRYAGAIN;
	  h_errno = TRY_AGAIN;
	  break;
	/* System has run out of file descriptors.  */
	case EMFILE:
	case ENFILE:
	  h_errno = NETDB_INTERNAL;
	  /* Fall through.  */
	case ECONNREFUSED:
	case ETIMEDOUT:
	  status = NSS_STATUS_UNAVAIL;
	  break;
	default:
	  status = NSS_STATUS_NOTFOUND;
	  break;
	}
      *h_errnop = h_errno;
      if (h_errno == TRY_AGAIN)
	*errnop = EAGAIN;
      else
	__set_errno (olderr);

      /* If we are looking for an IPv6 address and mapping is enabled
	 by having the RES_USE_INET6 bit in _res.options set, we try
	 another lookup.  */
      if (af == AF_INET6 && res_use_inet6 ())
	n = __res_context_search (ctx, name, C_IN, T_A, host_buffer.buf->buf,
				  host_buffer.buf != orig_host_buffer
				  ? MAXPACKET : 1024, &host_buffer.ptr,
				  NULL, NULL, NULL, NULL);

      if (n < 0)
	{
	  if (host_buffer.buf != orig_host_buffer)
	    free (host_buffer.buf);
	  return status;
	}

      map = 1;

      result->h_addrtype = AF_INET;
      result->h_length = INADDRSZ;
    }

  status = getanswer_r
    (ctx, host_buffer.buf, n, name, type, result, buffer, buflen,
     errnop, h_errnop, map, ttlp, canonp);
  if (host_buffer.buf != orig_host_buffer)
    free (host_buffer.buf);
  return status;
}

/* Verify that the name looks like a host name.  There is no point in
   sending a query which will not produce a usable name in the
   response.  */
static enum nss_status
check_name (const char *name, int *h_errnop)
{
  if (__libc_res_hnok (name))
    return NSS_STATUS_SUCCESS;
  *h_errnop = HOST_NOT_FOUND;
  return NSS_STATUS_NOTFOUND;
}

enum nss_status
_nss_dns_gethostbyname2_r (const char *name, int af, struct hostent *result,
			   char *buffer, size_t buflen, int *errnop,
			   int *h_errnop)
{
  enum nss_status status = check_name (name, h_errnop);
  if (status != NSS_STATUS_SUCCESS)
    return status;
  return _nss_dns_gethostbyname3_r (name, af, result, buffer, buflen, errnop,
				    h_errnop, NULL, NULL);
}
libc_hidden_def (_nss_dns_gethostbyname2_r)

enum nss_status
_nss_dns_gethostbyname_r (const char *name, struct hostent *result,
			  char *buffer, size_t buflen, int *errnop,
			  int *h_errnop)
{
  enum nss_status status = check_name (name, h_errnop);
  if (status != NSS_STATUS_SUCCESS)
    return status;
  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      *errnop = errno;
      *h_errnop = NETDB_INTERNAL;
      return NSS_STATUS_UNAVAIL;
    }
  status = NSS_STATUS_NOTFOUND;
  if (res_use_inet6 ())
    status = gethostbyname3_context (ctx, name, AF_INET6, result, buffer,
				     buflen, errnop, h_errnop, NULL, NULL);
  if (status == NSS_STATUS_NOTFOUND)
    status = gethostbyname3_context (ctx, name, AF_INET, result, buffer,
				     buflen, errnop, h_errnop, NULL, NULL);
  __resolv_context_put (ctx);
  return status;
}
libc_hidden_def (_nss_dns_gethostbyname_r)

enum nss_status
_nss_dns_gethostbyname4_r (const char *name, struct gaih_addrtuple **pat,
			   char *buffer, size_t buflen, int *errnop,
			   int *herrnop, int32_t *ttlp)
{
  enum nss_status status = check_name (name, herrnop);
  if (status != NSS_STATUS_SUCCESS)
    return status;
  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      *errnop = errno;
      *herrnop = NETDB_INTERNAL;
      return NSS_STATUS_UNAVAIL;
    }

  /*
   * if there aren't any dots, it could be a user-level alias.
   * this is also done in res_query() since we are not the only
   * function that looks up host names.
   */
  if (strchr (name, '.') == NULL)
    {
      char *tmp = alloca (NS_MAXDNAME);
      const char *cp = __res_context_hostalias (ctx, name, tmp, NS_MAXDNAME);
      if (cp != NULL)
	name = cp;
    }

  union
  {
    querybuf *buf;
    u_char *ptr;
  } host_buffer;
  querybuf *orig_host_buffer;
  host_buffer.buf = orig_host_buffer = (querybuf *) alloca (2048);
  u_char *ans2p = NULL;
  int nans2p = 0;
  int resplen2 = 0;
  int ans2p_malloced = 0;

  int olderr = errno;
  int n = __res_context_search (ctx, name, C_IN, T_QUERY_A_AND_AAAA,
				host_buffer.buf->buf, 2048, &host_buffer.ptr,
				&ans2p, &nans2p, &resplen2, &ans2p_malloced);
  if (n >= 0)
    {
      status = gaih_getanswer (host_buffer.buf, n, (const querybuf *) ans2p,
			       resplen2, name, pat, buffer, buflen,
			       errnop, herrnop, ttlp);
    }
  else
    {
      switch (errno)
	{
	case ESRCH:
	  status = NSS_STATUS_TRYAGAIN;
	  h_errno = TRY_AGAIN;
	  break;
	/* System has run out of file descriptors.  */
	case EMFILE:
	case ENFILE:
	  h_errno = NETDB_INTERNAL;
	  /* Fall through.  */
	case ECONNREFUSED:
	case ETIMEDOUT:
	  status = NSS_STATUS_UNAVAIL;
	  break;
	default:
	  status = NSS_STATUS_NOTFOUND;
	  break;
	}

      *herrnop = h_errno;
      if (h_errno == TRY_AGAIN)
	*errnop = EAGAIN;
      else
	__set_errno (olderr);
    }

  /* Check whether ans2p was separately allocated.  */
  if (ans2p_malloced)
    free (ans2p);

  if (host_buffer.buf != orig_host_buffer)
    free (host_buffer.buf);

  __resolv_context_put (ctx);
  return status;
}
libc_hidden_def (_nss_dns_gethostbyname4_r)

enum nss_status
_nss_dns_gethostbyaddr2_r (const void *addr, socklen_t len, int af,
			   struct hostent *result, char *buffer, size_t buflen,
			   int *errnop, int *h_errnop, int32_t *ttlp)
{
  static const u_char mapped[] = { 0,0, 0,0, 0,0, 0,0, 0,0, 0xff,0xff };
  static const u_char tunnelled[] = { 0,0, 0,0, 0,0, 0,0, 0,0, 0,0 };
  static const u_char v6local[] = { 0,0, 0,1 };
  const u_char *uaddr = (const u_char *)addr;
  struct host_data
  {
    char *aliases[MAX_NR_ALIASES];
    unsigned char host_addr[16];	/* IPv4 or IPv6 */
    char *h_addr_ptrs[MAX_NR_ADDRS + 1];
    char linebuffer[0];
  } *host_data = (struct host_data *) buffer;
  union
  {
    querybuf *buf;
    u_char *ptr;
  } host_buffer;
  querybuf *orig_host_buffer;
  char qbuf[MAXDNAME+1], *qp = NULL;
  size_t size;
  int n, status;
  int olderr = errno;

 uintptr_t pad = -(uintptr_t) buffer % __alignof__ (struct host_data);
 buffer += pad;
 buflen = buflen > pad ? buflen - pad : 0;

 if (__glibc_unlikely (buflen < sizeof (struct host_data)))
   {
     *errnop = ERANGE;
     *h_errnop = NETDB_INTERNAL;
     return NSS_STATUS_TRYAGAIN;
   }

 host_data = (struct host_data *) buffer;

  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      *errnop = errno;
      *h_errnop = NETDB_INTERNAL;
      return NSS_STATUS_UNAVAIL;
    }

  if (af == AF_INET6 && len == IN6ADDRSZ
      && (memcmp (uaddr, mapped, sizeof mapped) == 0
	  || (memcmp (uaddr, tunnelled, sizeof tunnelled) == 0
	      && memcmp (&uaddr[sizeof tunnelled], v6local, sizeof v6local))))
    {
      /* Unmap. */
      addr += sizeof mapped;
      uaddr += sizeof mapped;
      af = AF_INET;
      len = INADDRSZ;
    }

  switch (af)
    {
    case AF_INET:
      size = INADDRSZ;
      break;
    case AF_INET6:
      size = IN6ADDRSZ;
      break;
    default:
      *errnop = EAFNOSUPPORT;
      *h_errnop = NETDB_INTERNAL;
      __resolv_context_put (ctx);
      return NSS_STATUS_UNAVAIL;
    }
  if (size > len)
    {
      *errnop = EAFNOSUPPORT;
      *h_errnop = NETDB_INTERNAL;
      __resolv_context_put (ctx);
      return NSS_STATUS_UNAVAIL;
    }

  host_buffer.buf = orig_host_buffer = (querybuf *) alloca (1024);

  switch (af)
    {
    case AF_INET:
      sprintf (qbuf, "%u.%u.%u.%u.in-addr.arpa", (uaddr[3] & 0xff),
	       (uaddr[2] & 0xff), (uaddr[1] & 0xff), (uaddr[0] & 0xff));
      break;
    case AF_INET6:
      qp = qbuf;
      for (n = IN6ADDRSZ - 1; n >= 0; n--)
	{
	  static const char nibblechar[16] = "0123456789abcdef";
	  *qp++ = nibblechar[uaddr[n] & 0xf];
	  *qp++ = '.';
	  *qp++ = nibblechar[(uaddr[n] >> 4) & 0xf];
	  *qp++ = '.';
	}
      strcpy(qp, "ip6.arpa");
      break;
    default:
      /* Cannot happen.  */
      break;
    }

  n = __res_context_query (ctx, qbuf, C_IN, T_PTR, host_buffer.buf->buf,
			   1024, &host_buffer.ptr, NULL, NULL, NULL, NULL);
  if (n < 0)
    {
      *h_errnop = h_errno;
      __set_errno (olderr);
      if (host_buffer.buf != orig_host_buffer)
	free (host_buffer.buf);
      __resolv_context_put (ctx);
      return errno == ECONNREFUSED ? NSS_STATUS_UNAVAIL : NSS_STATUS_NOTFOUND;
    }

  status = getanswer_r
    (ctx, host_buffer.buf, n, qbuf, T_PTR, result, buffer, buflen,
     errnop, h_errnop, 0 /* XXX */, ttlp, NULL);
  if (host_buffer.buf != orig_host_buffer)
    free (host_buffer.buf);
  if (status != NSS_STATUS_SUCCESS)
    {
      __resolv_context_put (ctx);
      return status;
    }

  result->h_addrtype = af;
  result->h_length = len;
  memcpy (host_data->host_addr, addr, len);
  host_data->h_addr_ptrs[0] = (char *) host_data->host_addr;
  host_data->h_addr_ptrs[1] = NULL;
  *h_errnop = NETDB_SUCCESS;
  __resolv_context_put (ctx);
  return NSS_STATUS_SUCCESS;
}
libc_hidden_def (_nss_dns_gethostbyaddr2_r)


enum nss_status
_nss_dns_gethostbyaddr_r (const void *addr, socklen_t len, int af,
			  struct hostent *result, char *buffer, size_t buflen,
			  int *errnop, int *h_errnop)
{
  return _nss_dns_gethostbyaddr2_r (addr, len, af, result, buffer, buflen,
				    errnop, h_errnop, NULL);
}
libc_hidden_def (_nss_dns_gethostbyaddr_r)

static void
addrsort (struct resolv_context *ctx, char **ap, int num)
{
  int i, j;
  char **p;
  short aval[MAX_NR_ADDRS];
  int needsort = 0;
  size_t nsort = __resolv_context_sort_count (ctx);

  p = ap;
  if (num > MAX_NR_ADDRS)
    num = MAX_NR_ADDRS;
  for (i = 0; i < num; i++, p++)
    {
      for (j = 0 ; (unsigned)j < nsort; j++)
	{
	  struct resolv_sortlist_entry e
	    = __resolv_context_sort_entry (ctx, j);
	  if (e.addr.s_addr == (((struct in_addr *)(*p))->s_addr & e.mask))
	    break;
	}
      aval[i] = j;
      if (needsort == 0 && i > 0 && j < aval[i-1])
	needsort = i;
    }
  if (!needsort)
    return;

  while (needsort++ < num)
    for (j = needsort - 2; j >= 0; j--)
      if (aval[j] > aval[j+1])
	{
	  char *hp;

	  i = aval[j];
	  aval[j] = aval[j+1];
	  aval[j+1] = i;

	  hp = ap[j];
	  ap[j] = ap[j+1];
	  ap[j+1] = hp;
	}
      else
	break;
}

static enum nss_status
getanswer_r (struct resolv_context *ctx,
	     const querybuf *answer, int anslen, const char *qname, int qtype,
	     struct hostent *result, char *buffer, size_t buflen,
	     int *errnop, int *h_errnop, int map, int32_t *ttlp, char **canonp)
{
  struct host_data
  {
    char *aliases[MAX_NR_ALIASES];
    unsigned char host_addr[16];	/* IPv4 or IPv6 */
    char *h_addr_ptrs[0];
  } *host_data;
  int linebuflen;
  const HEADER *hp;
  const u_char *end_of_message, *cp;
  int n, ancount, qdcount;
  int haveanswer, had_error;
  char *bp, **ap, **hap;
  char tbuf[MAXDNAME];
  const char *tname;
  int (*name_ok) (const char *);
  u_char packtmp[NS_MAXCDNAME];
  int have_to_map = 0;
  uintptr_t pad = -(uintptr_t) buffer % __alignof__ (struct host_data);
  buffer += pad;
  buflen = buflen > pad ? buflen - pad : 0;
  if (__glibc_unlikely (buflen < sizeof (struct host_data)))
    {
      /* The buffer is too small.  */
    too_small:
      *errnop = ERANGE;
      *h_errnop = NETDB_INTERNAL;
      return NSS_STATUS_TRYAGAIN;
    }
  host_data = (struct host_data *) buffer;
  linebuflen = buflen - sizeof (struct host_data);
  if (buflen - sizeof (struct host_data) != linebuflen)
    linebuflen = INT_MAX;

  tname = qname;
  result->h_name = NULL;
  end_of_message = answer->buf + anslen;
  switch (qtype)
    {
    case T_A:
    case T_AAAA:
      name_ok = __libc_res_hnok;
      break;
    case T_PTR:
      name_ok = __libc_res_dnok;
      break;
    default:
      *errnop = ENOENT;
      return NSS_STATUS_UNAVAIL;  /* XXX should be abort(); */
    }

  /*
   * find first satisfactory answer
   */
  hp = &answer->hdr;
  ancount = ntohs (hp->ancount);
  qdcount = ntohs (hp->qdcount);
  cp = answer->buf + HFIXEDSZ;
  if (__glibc_unlikely (qdcount != 1))
    {
      *h_errnop = NO_RECOVERY;
      return NSS_STATUS_UNAVAIL;
    }
  if (sizeof (struct host_data) + (ancount + 1) * sizeof (char *) >= buflen)
    goto too_small;
  bp = (char *) &host_data->h_addr_ptrs[ancount + 1];
  linebuflen -= (ancount + 1) * sizeof (char *);

  n = __ns_name_unpack (answer->buf, end_of_message, cp,
			packtmp, sizeof packtmp);
  if (n != -1 && __ns_name_ntop (packtmp, bp, linebuflen) == -1)
    {
      if (__glibc_unlikely (errno == EMSGSIZE))
	goto too_small;

      n = -1;
    }

  if (__glibc_unlikely (n < 0))
    {
      *errnop = errno;
      *h_errnop = NO_RECOVERY;
      return NSS_STATUS_UNAVAIL;
    }
  if (__glibc_unlikely (name_ok (bp) == 0))
    {
      errno = EBADMSG;
      *errnop = EBADMSG;
      *h_errnop = NO_RECOVERY;
      return NSS_STATUS_UNAVAIL;
    }
  cp += n + QFIXEDSZ;

  if (qtype == T_A || qtype == T_AAAA)
    {
      /* res_send() has already verified that the query name is the
       * same as the one we sent; this just gets the expanded name
       * (i.e., with the succeeding search-domain tacked on).
       */
      n = strlen (bp) + 1;             /* for the \0 */
      if (n >= MAXHOSTNAMELEN)
	{
	  *h_errnop = NO_RECOVERY;
	  *errnop = ENOENT;
	  return NSS_STATUS_TRYAGAIN;
	}
      result->h_name = bp;
      bp += n;
      linebuflen -= n;
      if (linebuflen < 0)
	goto too_small;
      /* The qname can be abbreviated, but h_name is now absolute. */
      qname = result->h_name;
    }

  ap = host_data->aliases;
  *ap = NULL;
  result->h_aliases = host_data->aliases;
  hap = host_data->h_addr_ptrs;
  *hap = NULL;
  result->h_addr_list = host_data->h_addr_ptrs;
  haveanswer = 0;
  had_error = 0;

  while (ancount-- > 0 && cp < end_of_message && had_error == 0)
    {
      int type, class;

      n = __ns_name_unpack (answer->buf, end_of_message, cp,
			    packtmp, sizeof packtmp);
      if (n != -1 && __ns_name_ntop (packtmp, bp, linebuflen) == -1)
	{
	  if (__glibc_unlikely (errno == EMSGSIZE))
	    goto too_small;

	  n = -1;
	}

      if (__glibc_unlikely (n < 0 || (*name_ok) (bp) == 0))
	{
	  ++had_error;
	  continue;
	}
      cp += n;				/* name */

      if (__glibc_unlikely (cp + 10 > end_of_message))
	{
	  ++had_error;
	  continue;
	}

      NS_GET16 (type, cp);
      NS_GET16 (class, cp);
      int32_t ttl;
      NS_GET32 (ttl, cp);
      NS_GET16 (n, cp);		/* RDATA length.  */

      if (end_of_message - cp < n)
	{
	  /* RDATA extends beyond the end of the packet.  */
	  ++had_error;
	  continue;
	}

      if (__glibc_unlikely (class != C_IN))
	{
	  /* XXX - debug? syslog? */
	  cp += n;
	  continue;			/* XXX - had_error++ ? */
	}

      if ((qtype == T_A || qtype == T_AAAA) && type == T_CNAME)
	{
	  /* A CNAME could also have a TTL entry.  */
	  if (ttlp != NULL && ttl < *ttlp)
	      *ttlp = ttl;

	  if (ap >= &host_data->aliases[MAX_NR_ALIASES - 1])
	    continue;
	  n = __libc_dn_expand (answer->buf, end_of_message, cp,
				tbuf, sizeof tbuf);
	  if (__glibc_unlikely (n < 0 || (*name_ok) (tbuf) == 0))
	    {
	      ++had_error;
	      continue;
	    }
	  cp += n;
	  /* Store alias.  */
	  *ap++ = bp;
	  n = strlen (bp) + 1;		/* For the \0.  */
	  if (__glibc_unlikely (n >= MAXHOSTNAMELEN))
	    {
	      ++had_error;
	      continue;
	    }
	  bp += n;
	  linebuflen -= n;
	  /* Get canonical name.  */
	  n = strlen (tbuf) + 1;	/* For the \0.  */
	  if (__glibc_unlikely (n > linebuflen))
	    goto too_small;
	  if (__glibc_unlikely (n >= MAXHOSTNAMELEN))
	    {
	      ++had_error;
	      continue;
	    }
	  result->h_name = bp;
	  bp = __mempcpy (bp, tbuf, n);	/* Cannot overflow.  */
	  linebuflen -= n;
	  continue;
	}

      if (qtype == T_PTR && type == T_CNAME)
	{
	  /* A CNAME could also have a TTL entry.  */
	  if (ttlp != NULL && ttl < *ttlp)
	      *ttlp = ttl;

	  n = __libc_dn_expand (answer->buf, end_of_message, cp,
				tbuf, sizeof tbuf);
	  if (__glibc_unlikely (n < 0 || __libc_res_dnok (tbuf) == 0))
	    {
	      ++had_error;
	      continue;
	    }
	  cp += n;
	  /* Get canonical name.  */
	  n = strlen (tbuf) + 1;   /* For the \0.  */
	  if (__glibc_unlikely (n > linebuflen))
	    goto too_small;
	  if (__glibc_unlikely (n >= MAXHOSTNAMELEN))
	    {
	      ++had_error;
	      continue;
	    }
	  tname = bp;
	  bp = __mempcpy (bp, tbuf, n);	/* Cannot overflow.  */
	  linebuflen -= n;
	  continue;
	}

      if (type == T_A && qtype == T_AAAA && map)
	have_to_map = 1;
      else if (__glibc_unlikely (type != qtype))
	{
	  cp += n;
	  continue;			/* XXX - had_error++ ? */
	}

      switch (type)
	{
	case T_PTR:
	  if (__glibc_unlikely (__strcasecmp (tname, bp) != 0))
	    {
	      cp += n;
	      continue;			/* XXX - had_error++ ? */
	    }

	  n = __ns_name_unpack (answer->buf, end_of_message, cp,
				packtmp, sizeof packtmp);
	  if (n != -1 && __ns_name_ntop (packtmp, bp, linebuflen) == -1)
	    {
	      if (__glibc_unlikely (errno == EMSGSIZE))
		goto too_small;

	      n = -1;
	    }

	  if (__glibc_unlikely (n < 0 || __libc_res_hnok (bp) == 0))
	    {
	      ++had_error;
	      break;
	    }
	  if (ttlp != NULL && ttl < *ttlp)
	      *ttlp = ttl;
	  /* bind would put multiple PTR records as aliases, but we don't do
	     that.  */
	  result->h_name = bp;
	  *h_errnop = NETDB_SUCCESS;
	  return NSS_STATUS_SUCCESS;
	case T_A:
	case T_AAAA:
	  if (__glibc_unlikely (__strcasecmp (result->h_name, bp) != 0))
	    {
	      cp += n;
	      continue;			/* XXX - had_error++ ? */
	    }

	  /* Stop parsing at a record whose length is incorrect.  */
	  if (n != rrtype_to_rdata_length (type))
	    {
	      ++had_error;
	      break;
	    }

	  /* Skip records of the wrong type.  */
	  if (n != result->h_length)
	    {
	      cp += n;
	      continue;
	    }
	  if (!haveanswer)
	    {
	      int nn;

	      /* We compose a single hostent out of the entire chain of
	         entries, so the TTL of the hostent is essentially the lowest
		 TTL in the chain.  */
	      if (ttlp != NULL && ttl < *ttlp)
		*ttlp = ttl;
	      if (canonp != NULL)
		*canonp = bp;
	      result->h_name = bp;
	      nn = strlen (bp) + 1;	/* for the \0 */
	      bp += nn;
	      linebuflen -= nn;
	    }

	  /* Provide sufficient alignment for both address
	     families.  */
	  enum { align = 4 };
	  _Static_assert ((align % __alignof__ (struct in_addr)) == 0,
			  "struct in_addr alignment");
	  _Static_assert ((align % __alignof__ (struct in6_addr)) == 0,
			  "struct in6_addr alignment");
	  {
	    char *new_bp = PTR_ALIGN_UP (bp, align);
	    linebuflen -= new_bp - bp;
	    bp = new_bp;
	  }

	  if (__glibc_unlikely (n > linebuflen))
	    goto too_small;
	  bp = __mempcpy (*hap++ = bp, cp, n);
	  cp += n;
	  linebuflen -= n;
	  break;
	default:
	  abort ();
	}
      if (had_error == 0)
	++haveanswer;
    }

  if (haveanswer > 0)
    {
      *ap = NULL;
      *hap = NULL;
      /*
       * Note: we sort even if host can take only one address
       * in its return structures - should give it the "best"
       * address in that case, not some random one
       */
      if (haveanswer > 1 && qtype == T_A
	  && __resolv_context_sort_count (ctx) > 0)
	addrsort (ctx, host_data->h_addr_ptrs, haveanswer);

      if (result->h_name == NULL)
	{
	  n = strlen (qname) + 1;	/* For the \0.  */
	  if (n > linebuflen)
	    goto too_small;
	  if (n >= MAXHOSTNAMELEN)
	    goto no_recovery;
	  result->h_name = bp;
	  bp = __mempcpy (bp, qname, n);	/* Cannot overflow.  */
	  linebuflen -= n;
	}

      if (have_to_map)
	if (map_v4v6_hostent (result, &bp, &linebuflen))
	  goto too_small;
      *h_errnop = NETDB_SUCCESS;
      return NSS_STATUS_SUCCESS;
    }
 no_recovery:
  *h_errnop = NO_RECOVERY;
  *errnop = ENOENT;
  /* Special case here: if the resolver sent a result but it only
     contains a CNAME while we are looking for a T_A or T_AAAA record,
     we fail with NOTFOUND instead of TRYAGAIN.  */
  return ((qtype == T_A || qtype == T_AAAA) && ap != host_data->aliases
	   ? NSS_STATUS_NOTFOUND : NSS_STATUS_TRYAGAIN);
}


static enum nss_status
gaih_getanswer_slice (const querybuf *answer, int anslen, const char *qname,
		      struct gaih_addrtuple ***patp,
		      char **bufferp, size_t *buflenp,
		      int *errnop, int *h_errnop, int32_t *ttlp, int *firstp)
{
  char *buffer = *bufferp;
  size_t buflen = *buflenp;

  struct gaih_addrtuple **pat = *patp;
  const HEADER *hp = &answer->hdr;
  int ancount = ntohs (hp->ancount);
  int qdcount = ntohs (hp->qdcount);
  const u_char *cp = answer->buf + HFIXEDSZ;
  const u_char *end_of_message = answer->buf + anslen;
  if (__glibc_unlikely (qdcount != 1))
    {
      *h_errnop = NO_RECOVERY;
      return NSS_STATUS_UNAVAIL;
    }

  u_char packtmp[NS_MAXCDNAME];
  int n = __ns_name_unpack (answer->buf, end_of_message, cp,
			    packtmp, sizeof packtmp);
  /* We unpack the name to check it for validity.  But we do not need
     it later.  */
  if (n != -1 && __ns_name_ntop (packtmp, buffer, buflen) == -1)
    {
      if (__glibc_unlikely (errno == EMSGSIZE))
	{
	too_small:
	  *errnop = ERANGE;
	  *h_errnop = NETDB_INTERNAL;
	  return NSS_STATUS_TRYAGAIN;
	}

      n = -1;
    }

  if (__glibc_unlikely (n < 0))
    {
      *errnop = errno;
      *h_errnop = NO_RECOVERY;
      return NSS_STATUS_UNAVAIL;
    }
  if (__glibc_unlikely (__libc_res_hnok (buffer) == 0))
    {
      errno = EBADMSG;
      *errnop = EBADMSG;
      *h_errnop = NO_RECOVERY;
      return NSS_STATUS_UNAVAIL;
    }
  cp += n + QFIXEDSZ;

  int haveanswer = 0;
  int had_error = 0;
  char *canon = NULL;
  char *h_name = NULL;
  int h_namelen = 0;

  if (ancount == 0)
    {
      *h_errnop = HOST_NOT_FOUND;
      return NSS_STATUS_NOTFOUND;
    }

  while (ancount-- > 0 && cp < end_of_message && had_error == 0)
    {
      n = __ns_name_unpack (answer->buf, end_of_message, cp,
			    packtmp, sizeof packtmp);
      if (n != -1 &&
	  (h_namelen = __ns_name_ntop (packtmp, buffer, buflen)) == -1)
	{
	  if (__glibc_unlikely (errno == EMSGSIZE))
	    goto too_small;

	  n = -1;
	}
      if (__glibc_unlikely (n < 0 || __libc_res_hnok (buffer) == 0))
	{
	  ++had_error;
	  continue;
	}
      if (*firstp && canon == NULL)
	{
	  h_name = buffer;
	  buffer += h_namelen;
	  buflen -= h_namelen;
	}

      cp += n;				/* name */

      if (__glibc_unlikely (cp + 10 > end_of_message))
	{
	  ++had_error;
	  continue;
	}

      uint16_t type;
      NS_GET16 (type, cp);
      uint16_t class;
      NS_GET16 (class, cp);
      int32_t ttl;
      NS_GET32 (ttl, cp);
      NS_GET16 (n, cp);		/* RDATA length.  */

      if (end_of_message - cp < n)
	{
	  /* RDATA extends beyond the end of the packet.  */
	  ++had_error;
	  continue;
	}

      if (class != C_IN)
	{
	  cp += n;
	  continue;
	}

      if (type == T_CNAME)
	{
	  char tbuf[MAXDNAME];

	  /* A CNAME could also have a TTL entry.  */
	  if (ttlp != NULL && ttl < *ttlp)
	      *ttlp = ttl;

	  n = __libc_dn_expand (answer->buf, end_of_message, cp,
				tbuf, sizeof tbuf);
	  if (__glibc_unlikely (n < 0 || __libc_res_hnok (tbuf) == 0))
	    {
	      ++had_error;
	      continue;
	    }
	  cp += n;

	  if (*firstp)
	    {
	      /* Reclaim buffer space.  */
	      if (h_name + h_namelen == buffer)
		{
		  buffer = h_name;
		  buflen += h_namelen;
		}

	      n = strlen (tbuf) + 1;
	      if (__glibc_unlikely (n > buflen))
		goto too_small;
	      if (__glibc_unlikely (n >= MAXHOSTNAMELEN))
		{
		  ++had_error;
		  continue;
		}

	      canon = buffer;
	      buffer = __mempcpy (buffer, tbuf, n);
	      buflen -= n;
	      h_namelen = 0;
	    }
	  continue;
	}

      /* Stop parsing if we encounter a record with incorrect RDATA
	 length.  */
      if (type == T_A || type == T_AAAA)
	{
	  if (n != rrtype_to_rdata_length (type))
	    {
	      ++had_error;
	      continue;
	    }
	}
      else
	{
	  /* Skip unknown records.  */
	  cp += n;
	  continue;
	}

      assert (type == T_A || type == T_AAAA);
      if (*pat == NULL)
	{
	  uintptr_t pad = (-(uintptr_t) buffer
			   % __alignof__ (struct gaih_addrtuple));
	  buffer += pad;
	  buflen = buflen > pad ? buflen - pad : 0;

	  if (__glibc_unlikely (buflen < sizeof (struct gaih_addrtuple)))
	    goto too_small;

	  *pat = (struct gaih_addrtuple *) buffer;
	  buffer += sizeof (struct gaih_addrtuple);
	  buflen -= sizeof (struct gaih_addrtuple);
	}

      (*pat)->name = NULL;
      (*pat)->next = NULL;

      if (*firstp)
	{
	  /* We compose a single hostent out of the entire chain of
	     entries, so the TTL of the hostent is essentially the lowest
	     TTL in the chain.  */
	  if (ttlp != NULL && ttl < *ttlp)
	    *ttlp = ttl;

	  (*pat)->name = canon ?: h_name;

	  *firstp = 0;
	}

      (*pat)->family = type == T_A ? AF_INET : AF_INET6;
      memcpy ((*pat)->addr, cp, n);
      cp += n;
      (*pat)->scopeid = 0;

      pat = &((*pat)->next);

      haveanswer = 1;
    }

  if (haveanswer)
    {
      *patp = pat;
      *bufferp = buffer;
      *buflenp = buflen;

      *h_errnop = NETDB_SUCCESS;
      return NSS_STATUS_SUCCESS;
    }

  /* Special case here: if the resolver sent a result but it only
     contains a CNAME while we are looking for a T_A or T_AAAA record,
     we fail with NOTFOUND instead of TRYAGAIN.  */
  if (canon != NULL)
    {
      *h_errnop = HOST_NOT_FOUND;
      return NSS_STATUS_NOTFOUND;
    }

  *h_errnop = NETDB_INTERNAL;
  return NSS_STATUS_TRYAGAIN;
}


static enum nss_status
gaih_getanswer (const querybuf *answer1, int anslen1, const querybuf *answer2,
		int anslen2, const char *qname,
		struct gaih_addrtuple **pat, char *buffer, size_t buflen,
		int *errnop, int *h_errnop, int32_t *ttlp)
{
  int first = 1;

  enum nss_status status = NSS_STATUS_NOTFOUND;

  /* Combining the NSS status of two distinct queries requires some
     compromise and attention to symmetry (A or AAAA queries can be
     returned in any order).  What follows is a breakdown of how this
     code is expected to work and why. We discuss only SUCCESS,
     TRYAGAIN, NOTFOUND and UNAVAIL, since they are the only returns
     that apply (though RETURN and MERGE exist).  We make a distinction
     between TRYAGAIN (recoverable) and TRYAGAIN' (not-recoverable).
     A recoverable TRYAGAIN is almost always due to buffer size issues
     and returns ERANGE in errno and the caller is expected to retry
     with a larger buffer.

     Lastly, you may be tempted to make significant changes to the
     conditions in this code to bring about symmetry between responses.
     Please don't change anything without due consideration for
     expected application behaviour.  Some of the synthesized responses
     aren't very well thought out and sometimes appear to imply that
     IPv4 responses are always answer 1, and IPv6 responses are always
     answer 2, but that's not true (see the implementation of send_dg
     and send_vc to see response can arrive in any order, particularly
     for UDP). However, we expect it holds roughly enough of the time
     that this code works, but certainly needs to be fixed to make this
     a more robust implementation.

     ----------------------------------------------
     | Answer 1 Status /   | Synthesized | Reason |
     | Answer 2 Status     | Status      |        |
     |--------------------------------------------|
     | SUCCESS/SUCCESS     | SUCCESS     | [1]    |
     | SUCCESS/TRYAGAIN    | TRYAGAIN    | [5]    |
     | SUCCESS/TRYAGAIN'   | SUCCESS     | [1]    |
     | SUCCESS/NOTFOUND    | SUCCESS     | [1]    |
     | SUCCESS/UNAVAIL     | SUCCESS     | [1]    |
     | TRYAGAIN/SUCCESS    | TRYAGAIN    | [2]    |
     | TRYAGAIN/TRYAGAIN   | TRYAGAIN    | [2]    |
     | TRYAGAIN/TRYAGAIN'  | TRYAGAIN    | [2]    |
     | TRYAGAIN/NOTFOUND   | TRYAGAIN    | [2]    |
     | TRYAGAIN/UNAVAIL    | TRYAGAIN    | [2]    |
     | TRYAGAIN'/SUCCESS   | SUCCESS     | [3]    |
     | TRYAGAIN'/TRYAGAIN  | TRYAGAIN    | [3]    |
     | TRYAGAIN'/TRYAGAIN' | TRYAGAIN'   | [3]    |
     | TRYAGAIN'/NOTFOUND  | TRYAGAIN'   | [3]    |
     | TRYAGAIN'/UNAVAIL   | UNAVAIL     | [3]    |
     | NOTFOUND/SUCCESS    | SUCCESS     | [3]    |
     | NOTFOUND/TRYAGAIN   | TRYAGAIN    | [3]    |
     | NOTFOUND/TRYAGAIN'  | TRYAGAIN'   | [3]    |
     | NOTFOUND/NOTFOUND   | NOTFOUND    | [3]    |
     | NOTFOUND/UNAVAIL    | UNAVAIL     | [3]    |
     | UNAVAIL/SUCCESS     | UNAVAIL     | [4]    |
     | UNAVAIL/TRYAGAIN    | UNAVAIL     | [4]    |
     | UNAVAIL/TRYAGAIN'   | UNAVAIL     | [4]    |
     | UNAVAIL/NOTFOUND    | UNAVAIL     | [4]    |
     | UNAVAIL/UNAVAIL     | UNAVAIL     | [4]    |
     ----------------------------------------------

     [1] If the first response is a success we return success.
	 This ignores the state of the second answer and in fact
	 incorrectly sets errno and h_errno to that of the second
	 answer.  However because the response is a success we ignore
	 *errnop and *h_errnop (though that means you touched errno on
	 success).  We are being conservative here and returning the
	 likely IPv4 response in the first answer as a success.

     [2] If the first response is a recoverable TRYAGAIN we return
	 that instead of looking at the second response.  The
	 expectation here is that we have failed to get an IPv4 response
	 and should retry both queries.

     [3] If the first response was not a SUCCESS and the second
	 response is not NOTFOUND (had a SUCCESS, need to TRYAGAIN,
	 or failed entirely e.g. TRYAGAIN' and UNAVAIL) then use the
	 result from the second response, otherwise the first responses
	 status is used.  Again we have some odd side-effects when the
	 second response is NOTFOUND because we overwrite *errnop and
	 *h_errnop that means that a first answer of NOTFOUND might see
	 its *errnop and *h_errnop values altered.  Whether it matters
	 in practice that a first response NOTFOUND has the wrong
	 *errnop and *h_errnop is undecided.

     [4] If the first response is UNAVAIL we return that instead of
	 looking at the second response.  The expectation here is that
	 it will have failed similarly e.g. configuration failure.

     [5] Testing this code is complicated by the fact that truncated
	 second response buffers might be returned as SUCCESS if the
	 first answer is a SUCCESS.  To fix this we add symmetry to
	 TRYAGAIN with the second response.  If the second response
	 is a recoverable error we now return TRYAGIN even if the first
	 response was SUCCESS.  */

  if (anslen1 > 0)
    status = gaih_getanswer_slice(answer1, anslen1, qname,
				  &pat, &buffer, &buflen,
				  errnop, h_errnop, ttlp,
				  &first);

  if ((status == NSS_STATUS_SUCCESS || status == NSS_STATUS_NOTFOUND
       || (status == NSS_STATUS_TRYAGAIN
	   /* We want to look at the second answer in case of an
	      NSS_STATUS_TRYAGAIN only if the error is non-recoverable, i.e.
	      *h_errnop is NO_RECOVERY. If not, and if the failure was due to
	      an insufficient buffer (ERANGE), then we need to drop the results
	      and pass on the NSS_STATUS_TRYAGAIN to the caller so that it can
	      repeat the query with a larger buffer.  */
	   && (*errnop != ERANGE || *h_errnop == NO_RECOVERY)))
      && answer2 != NULL && anslen2 > 0)
    {
      enum nss_status status2 = gaih_getanswer_slice(answer2, anslen2, qname,
						     &pat, &buffer, &buflen,
						     errnop, h_errnop, ttlp,
						     &first);
      /* Use the second response status in some cases.  */
      if (status != NSS_STATUS_SUCCESS && status2 != NSS_STATUS_NOTFOUND)
	status = status2;
      /* Do not return a truncated second response (unless it was
	 unavoidable e.g. unrecoverable TRYAGAIN).  */
      if (status == NSS_STATUS_SUCCESS
	  && (status2 == NSS_STATUS_TRYAGAIN
	      && *errnop == ERANGE && *h_errnop != NO_RECOVERY))
	status = NSS_STATUS_TRYAGAIN;
    }

  return status;
}
