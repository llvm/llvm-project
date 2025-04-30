/* Creation of DNS query packets.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
 * Copyright (c) 1985, 1993
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

#include <stdint.h>
#include <sys/types.h>
#include <sys/param.h>
#include <netinet/in.h>
#include <arpa/nameser.h>
#include <netdb.h>
#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>
#include <string.h>
#include <sys/time.h>
#include <shlib-compat.h>
#include <random-bits.h>

int
__res_context_mkquery (struct resolv_context *ctx, int op, const char *dname,
                       int class, int type, const unsigned char *data,
                       unsigned char *buf, int buflen)
{
  HEADER *hp;
  unsigned char *cp;
  int n;
  unsigned char *dnptrs[20], **dpp, **lastdnptr;

  if (class < 0 || class > 65535 || type < 0 || type > 65535)
    return -1;

  /* Initialize header fields.  */
  if ((buf == NULL) || (buflen < HFIXEDSZ))
    return -1;
  memset (buf, 0, HFIXEDSZ);
  hp = (HEADER *) buf;
  /* We randomize the IDs every time.  The old code just incremented
     by one after the initial randomization which still predictable if
     the application does multiple requests.  */
  hp->id = random_bits ();
  hp->opcode = op;
  if (ctx->resp->options & RES_TRUSTAD)
    hp->ad = 1;
  hp->rd = (ctx->resp->options & RES_RECURSE) != 0;
  hp->rcode = NOERROR;
  cp = buf + HFIXEDSZ;
  buflen -= HFIXEDSZ;
  dpp = dnptrs;
  *dpp++ = buf;
  *dpp++ = NULL;
  lastdnptr = dnptrs + sizeof dnptrs / sizeof dnptrs[0];

  /* Perform opcode specific processing.  */
  switch (op)
    {
    case NS_NOTIFY_OP:
      if ((buflen -= QFIXEDSZ + (data == NULL ? 0 : RRFIXEDSZ)) < 0)
        return -1;
      goto compose;

    case QUERY:
      if ((buflen -= QFIXEDSZ) < 0)
        return -1;
    compose:
      n = __ns_name_compress (dname, cp, buflen,
                              (const unsigned char **) dnptrs,
                              (const unsigned char **) lastdnptr);
      if (n < 0)
        return -1;
      cp += n;
      buflen -= n;
      NS_PUT16 (type, cp);
      NS_PUT16 (class, cp);
      hp->qdcount = htons (1);
      if (op == QUERY || data == NULL)
        break;

      /* Make an additional record for completion domain.  */
      n = __ns_name_compress ((char *)data, cp, buflen,
                              (const unsigned char **) dnptrs,
                              (const unsigned char **) lastdnptr);
      if (__glibc_unlikely (n < 0))
        return -1;
      cp += n;
      buflen -= n;
      NS_PUT16 (T_NULL, cp);
      NS_PUT16 (class, cp);
      NS_PUT32 (0, cp);
      NS_PUT16 (0, cp);
      hp->arcount = htons (1);
      break;

    default:
      return -1;
    }
  return cp - buf;
}
libc_hidden_def (__res_context_mkquery)

/* Common part of res_nmkquery and res_mkquery.  */
static int
context_mkquery_common (struct resolv_context *ctx,
                        int op, const char *dname, int class, int type,
                        const unsigned char *data,
                        unsigned char *buf, int buflen)
{
  if (ctx == NULL)
    return -1;
  int result = __res_context_mkquery
    (ctx, op, dname, class, type, data, buf, buflen);
  if (result >= 2)
    memcpy (&ctx->resp->id, buf, 2);
  __resolv_context_put (ctx);
  return result;
}

/* Form all types of queries.  Returns the size of the result or -1 on
   error.

   STATP points to an initialized resolver state.  OP is the opcode of
   the query.  DNAME is the domain.  CLASS and TYPE are the DNS query
   class and type.  DATA can be NULL; otherwise, it is a pointer to a
   domain name which is included in the generated packet (if op ==
   NS_NOTIFY_OP).  BUF must point to the out buffer of BUFLEN bytes.

   DATALEN and NEWRR_IN are currently ignored.  */
int
___res_nmkquery (res_state statp, int op, const char *dname,
                 int class, int type,
                 const unsigned char *data, int datalen,
                 const unsigned char *newrr_in,
                 unsigned char *buf, int buflen)
{
  return context_mkquery_common
    (__resolv_context_get_override (statp),
     op, dname, class, type, data, buf, buflen);
}
versioned_symbol (libc, ___res_nmkquery, res_nmkquery, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_nmkquery, __res_nmkquery, GLIBC_2_2);
#endif

int
___res_mkquery (int op, const char *dname, int class, int type,
                const unsigned char *data, int datalen,
                const unsigned char *newrr_in,
                unsigned char *buf, int buflen)
{
  return context_mkquery_common
    (__resolv_context_get_preinit (),
     op, dname, class, type, data, buf, buflen);
}
versioned_symbol (libc, ___res_mkquery, res_mkquery, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_2)
compat_symbol (libresolv, ___res_mkquery, res_mkquery, GLIBC_2_0);
#endif
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libresolv, ___res_mkquery, __res_mkquery, GLIBC_2_2);
#endif

/* Create an OPT resource record.  Return the length of the final
   packet, or -1 on error.

   STATP must be an initialized resolver state.  N0 is the current
   number of bytes of the packet (already written to BUF by the
   aller).  BUF is the packet being constructed.  The array it
   pointers to must be BUFLEN bytes long.  ANSLEN is the advertised
   EDNS buffer size (to be included in the OPT resource record).  */
int
__res_nopt (struct resolv_context *ctx,
            int n0, unsigned char *buf, int buflen, int anslen)
{
  uint16_t flags = 0;
  HEADER *hp = (HEADER *) buf;
  unsigned char *cp = buf + n0;
  unsigned char *ep = buf + buflen;

  if ((ep - cp) < 1 + RRFIXEDSZ)
    return -1;

  /* Add the root label.  */
  *cp++ = 0;

  NS_PUT16 (T_OPT, cp);         /* Record type.  */

  /* Lowering the advertised buffer size based on the actual
     answer buffer size is desirable because the server will
     minimize the reply to fit into the UDP packet (and A
     non-minimal response might not fit the buffer).

     The RESOLV_EDNS_BUFFER_SIZE limit could still result in TCP
     fallback and a non-minimal response which has to be
     hard-truncated in the stub resolver, but this is price to
     pay for avoiding fragmentation.  (This issue does not
     affect the nss_dns functions because they use the stub
     resolver in such a way that it allocates a properly sized
     response buffer.)  */
  {
    uint16_t buffer_size;
    if (anslen < 512)
      buffer_size = 512;
    else if (anslen > RESOLV_EDNS_BUFFER_SIZE)
      buffer_size = RESOLV_EDNS_BUFFER_SIZE;
    else
      buffer_size = anslen;
    NS_PUT16 (buffer_size, cp);
  }

  *cp++ = NOERROR;              /* Extended RCODE.  */
  *cp++ = 0;                    /* EDNS version.  */

  if (ctx->resp->options & RES_USE_DNSSEC)
    flags |= NS_OPT_DNSSEC_OK;

  NS_PUT16 (flags, cp);
  NS_PUT16 (0, cp);       /* RDATA length (no options are preent).  */
  hp->arcount = htons (ntohs (hp->arcount) + 1);

  return cp - buf;
}
libc_hidden_def (__res_nopt)
