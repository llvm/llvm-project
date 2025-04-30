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

/* Parts of this file are plain copies of the file `getnetnamadr.c' from
   the bind package and it has the following copyright.  */

/* Copyright (c) 1993 Carlos Leandro and Rui Salgueiro
 *      Dep. Matematica Universidade de Coimbra, Portugal, Europe
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 */
/*
 * Copyright (c) 1983, 1993
 *      The Regents of the University of California.  All rights reserved.
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

#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "nsswitch.h"
#include <arpa/inet.h>
#include <arpa/nameser.h>
#include <nss_dns.h>
#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>

/* Maximum number of aliases we allow.  */
#define MAX_NR_ALIASES	48


#if PACKETSZ > 65536
# define MAXPACKET	PACKETSZ
#else
# define MAXPACKET	65536
#endif


typedef enum
{
  BYADDR,
  BYNAME
} lookup_method;


/* We need this time later.  */
typedef union querybuf
{
  HEADER hdr;
  u_char buf[MAXPACKET];
} querybuf;

/* Prototypes for local functions.  */
static enum nss_status getanswer_r (const querybuf *answer, int anslen,
				    struct netent *result, char *buffer,
				    size_t buflen, int *errnop, int *h_errnop,
				    lookup_method net_i);


enum nss_status
_nss_dns_getnetbyname_r (const char *name, struct netent *result,
			 char *buffer, size_t buflen, int *errnop,
			 int *herrnop)
{
  /* Return entry for network with NAME.  */
  union
  {
    querybuf *buf;
    u_char *ptr;
  } net_buffer;
  querybuf *orig_net_buffer;
  int anslen;
  enum nss_status status;

  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      *errnop = errno;
      *herrnop = NETDB_INTERNAL;
      return NSS_STATUS_UNAVAIL;
    }

  net_buffer.buf = orig_net_buffer = (querybuf *) alloca (1024);

  anslen = __res_context_search
    (ctx, name, C_IN, T_PTR, net_buffer.buf->buf,
     1024, &net_buffer.ptr, NULL, NULL, NULL, NULL);
  if (anslen < 0)
    {
      /* Nothing found.  */
      *errnop = errno;
      if (net_buffer.buf != orig_net_buffer)
	free (net_buffer.buf);
      __resolv_context_put (ctx);
      return (errno == ECONNREFUSED
	      || errno == EPFNOSUPPORT
	      || errno == EAFNOSUPPORT)
	? NSS_STATUS_UNAVAIL : NSS_STATUS_NOTFOUND;
    }

  status = getanswer_r (net_buffer.buf, anslen, result, buffer, buflen,
			errnop, herrnop, BYNAME);
  if (net_buffer.buf != orig_net_buffer)
    free (net_buffer.buf);
  __resolv_context_put (ctx);
  return status;
}
libc_hidden_def (_nss_dns_getnetbyname_r)

enum nss_status
_nss_dns_getnetbyaddr_r (uint32_t net, int type, struct netent *result,
			 char *buffer, size_t buflen, int *errnop,
			 int *herrnop)
{
  /* Return entry for network with NAME.  */
  enum nss_status status;
  union
  {
    querybuf *buf;
    u_char *ptr;
  } net_buffer;
  querybuf *orig_net_buffer;
  unsigned int net_bytes[4];
  char qbuf[MAXDNAME];
  int cnt, anslen;
  uint32_t net2;
  int olderr = errno;

  /* No net address lookup for IPv6 yet.  */
  if (type != AF_INET)
    return NSS_STATUS_UNAVAIL;

  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      *errnop = errno;
      *herrnop = NETDB_INTERNAL;
      return NSS_STATUS_UNAVAIL;
    }

  net2 = (uint32_t) net;
  for (cnt = 4; net2 != 0; net2 >>= 8)
    net_bytes[--cnt] = net2 & 0xff;

  switch (cnt)
    {
    case 3:
      /* Class A network.  */
      sprintf (qbuf, "0.0.0.%u.in-addr.arpa", net_bytes[3]);
      break;
    case 2:
      /* Class B network.  */
      sprintf (qbuf, "0.0.%u.%u.in-addr.arpa", net_bytes[3], net_bytes[2]);
      break;
    case 1:
      /* Class C network.  */
      sprintf (qbuf, "0.%u.%u.%u.in-addr.arpa", net_bytes[3], net_bytes[2],
	       net_bytes[1]);
      break;
    case 0:
      /* Class D - E network.  */
      sprintf (qbuf, "%u.%u.%u.%u.in-addr.arpa", net_bytes[3], net_bytes[2],
	       net_bytes[1], net_bytes[0]);
      break;
    }

  net_buffer.buf = orig_net_buffer = (querybuf *) alloca (1024);

  anslen = __res_context_query (ctx, qbuf, C_IN, T_PTR, net_buffer.buf->buf,
				1024, &net_buffer.ptr, NULL, NULL, NULL, NULL);
  if (anslen < 0)
    {
      /* Nothing found.  */
      int err = errno;
      __set_errno (olderr);
      if (net_buffer.buf != orig_net_buffer)
	free (net_buffer.buf);
      __resolv_context_put (ctx);
      return (err == ECONNREFUSED
	      || err == EPFNOSUPPORT
	      || err == EAFNOSUPPORT)
	? NSS_STATUS_UNAVAIL : NSS_STATUS_NOTFOUND;
    }

  status = getanswer_r (net_buffer.buf, anslen, result, buffer, buflen,
			errnop, herrnop, BYADDR);
  if (net_buffer.buf != orig_net_buffer)
    free (net_buffer.buf);
  if (status == NSS_STATUS_SUCCESS)
    {
      /* Strip trailing zeros.  */
      unsigned int u_net = net;	/* Maybe net should be unsigned?  */

      while ((u_net & 0xff) == 0 && u_net != 0)
	u_net >>= 8;
      result->n_net = u_net;
    }

  __resolv_context_put (ctx);
  return status;
}
libc_hidden_def (_nss_dns_getnetbyaddr_r)

static enum nss_status
getanswer_r (const querybuf *answer, int anslen, struct netent *result,
	     char *buffer, size_t buflen, int *errnop, int *h_errnop,
	     lookup_method net_i)
{
  /*
   * Find first satisfactory answer
   *
   *      answer --> +------------+  ( MESSAGE )
   *                 |   Header   |
   *                 +------------+
   *                 |  Question  | the question for the name server
   *                 +------------+
   *                 |   Answer   | RRs answering the question
   *                 +------------+
   *                 | Authority  | RRs pointing toward an authority
   *                 | Additional | RRs holding additional information
   *                 +------------+
   */
  struct net_data
  {
    char *aliases[MAX_NR_ALIASES];
    char linebuffer[0];
  } *net_data;

  uintptr_t pad = -(uintptr_t) buffer % __alignof__ (struct net_data);
  buffer += pad;

  if (__glibc_unlikely (buflen < sizeof (*net_data) + pad))
    {
      /* The buffer is too small.  */
    too_small:
      *errnop = ERANGE;
      *h_errnop = NETDB_INTERNAL;
      return NSS_STATUS_TRYAGAIN;
    }
  buflen -= pad;

  net_data = (struct net_data *) buffer;
  int linebuflen = buflen - offsetof (struct net_data, linebuffer);
  if (buflen - offsetof (struct net_data, linebuffer) != linebuflen)
    linebuflen = INT_MAX;
  const unsigned char *end_of_message = &answer->buf[anslen];
  const HEADER *header_pointer = &answer->hdr;
  /* #/records in the answer section.  */
  int answer_count =  ntohs (header_pointer->ancount);
  /* #/entries in the question section.  */
  int question_count = ntohs (header_pointer->qdcount);
  char *bp = net_data->linebuffer;
  const unsigned char *cp = &answer->buf[HFIXEDSZ];
  char **alias_pointer;
  int have_answer;
  u_char packtmp[NS_MAXCDNAME];

  if (question_count == 0)
    {
      /* FIXME: the Sun version uses for host name lookup an additional
	 parameter for pointing to h_errno.  this is missing here.
	 OSF/1 has a per-thread h_errno variable.  */
      if (header_pointer->aa != 0)
	{
	  __set_h_errno (HOST_NOT_FOUND);
	  return NSS_STATUS_NOTFOUND;
	}
      else
	{
	  __set_h_errno (TRY_AGAIN);
	  return NSS_STATUS_TRYAGAIN;
	}
    }

  /* Skip the question part.  */
  while (question_count-- > 0)
    {
      int n = __libc_dn_skipname (cp, end_of_message);
      if (n < 0 || end_of_message - (cp + n) < QFIXEDSZ)
       {
         __set_h_errno (NO_RECOVERY);
         return NSS_STATUS_UNAVAIL;
       }
      cp += n + QFIXEDSZ;
    }

  alias_pointer = result->n_aliases = &net_data->aliases[0];
  *alias_pointer = NULL;
  have_answer = 0;

  while (--answer_count >= 0 && cp < end_of_message)
    {
      int n = __ns_name_unpack (answer->buf, end_of_message, cp,
				packtmp, sizeof packtmp);
      if (n != -1 && __ns_name_ntop (packtmp, bp, linebuflen) == -1)
	{
	  if (errno == EMSGSIZE)
	    goto too_small;

	  n = -1;
	}

      if (n < 0 || __libc_res_dnok (bp) == 0)
	break;
      cp += n;

      if (end_of_message - cp < 10)
	{
	  __set_h_errno (NO_RECOVERY);
	  return NSS_STATUS_UNAVAIL;
	}

      int type, class;
      GETSHORT (type, cp);
      GETSHORT (class, cp);
      cp += INT32SZ;		/* TTL */
      uint16_t rdatalen;
      GETSHORT (rdatalen, cp);
      if (end_of_message - cp < rdatalen)
	{
	  __set_h_errno (NO_RECOVERY);
	  return NSS_STATUS_UNAVAIL;
	}

      if (class == C_IN && type == T_PTR)
	{
	  n = __ns_name_unpack (answer->buf, end_of_message, cp,
				packtmp, sizeof packtmp);
	  if (n != -1 && __ns_name_ntop (packtmp, bp, linebuflen) == -1)
	    {
	      if (errno == EMSGSIZE)
		goto too_small;

	      n = -1;
	    }

	  if (n < 0 || !__libc_res_hnok (bp))
	    {
	      /* XXX What does this mean?  The original form from bind
		 returns NULL. Incrementing cp has no effect in any case.
		 What should I return here. ??? */
	      cp += n;
	      return NSS_STATUS_UNAVAIL;
	    }
	  cp += rdatalen;
         if (alias_pointer + 2 < &net_data->aliases[MAX_NR_ALIASES])
           {
             *alias_pointer++ = bp;
             n = strlen (bp) + 1;
             bp += n;
             linebuflen -= n;
             result->n_addrtype = class == C_IN ? AF_INET : AF_UNSPEC;
             ++have_answer;
           }
	}
      else
	/* Skip over unknown record data.  */
	cp += rdatalen;
    }

  if (have_answer)
    {
      *alias_pointer = NULL;
      switch (net_i)
	{
	case BYADDR:
	  result->n_name = *result->n_aliases++;
	  result->n_net = 0L;
	  return NSS_STATUS_SUCCESS;

	case BYNAME:
	  {
	    char **ap;
	    for (ap = result->n_aliases; *ap != NULL; ++ap)
	      {
		/* Check each alias name for being of the forms:
		   4.3.2.1.in-addr.arpa		= net 1.2.3.4
		   3.2.1.in-addr.arpa		= net 0.1.2.3
		   2.1.in-addr.arpa		= net 0.0.1.2
		   1.in-addr.arpa		= net 0.0.0.1
		*/
		uint32_t val = 0;	/* Accumulator for n_net value.  */
		unsigned int shift = 0; /* Which part we are parsing now.  */
		const char *p = *ap; /* Consuming the string.  */
		do
		  {
		    /* Match the leading 0 or 0[xX] base indicator.  */
		    unsigned int base = 10;
		    if (*p == '0' && p[1] != '.')
		      {
			base = 8;
			++p;
			if (*p == 'x' || *p == 'X')
			  {
			    base = 16;
			    ++p;
			    if (*p == '.')
			      break; /* No digit here.  Give up on alias.  */
			  }
			if (*p == '\0')
			  break;
		      }

		    uint32_t part = 0; /* Accumulates this part's number.  */
		    do
		      {
			if (isdigit (*p) && (*p - '0' < base))
			  part = (part * base) + (*p - '0');
			else if (base == 16 && isxdigit (*p))
			  part = (part << 4) + 10 + (tolower (*p) - 'a');
			++p;
		      } while (*p != '\0' && *p != '.');

		    if (*p != '.')
		      break;	/* Bad form.  Give up on this name.  */

		    /* Install this as the next more significant byte.  */
		    val |= part << shift;
		    shift += 8;
		    ++p;

		    /* If we are out of digits now, there are two cases:
		       1. We are done with digits and now see "in-addr.arpa".
		       2. This is not the droid we are looking for.  */
		    if (!isdigit (*p) && !__strcasecmp (p, "in-addr.arpa"))
		      {
			result->n_net = val;
			return NSS_STATUS_SUCCESS;
		      }

		    /* Keep going when we have seen fewer than 4 parts.  */
		  } while (shift < 32);
	      }
	  }
	  break;
	}
    }

  __set_h_errno (TRY_AGAIN);
  return NSS_STATUS_TRYAGAIN;
}
