/* Legacy IPv4 text-to-address functions.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
 * Copyright (c) 1983, 1990, 1993
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

#include <sys/types.h>
#include <sys/param.h>

#include <netinet/in.h>
#include <arpa/inet.h>

#include <ctype.h>

#include <endian.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <errno.h>

/* Check whether "cp" is a valid ASCII representation of an IPv4
   Internet address and convert it to a binary address.  Returns 1 if
   the address is valid, 0 if not.  This replaces inet_addr, the
   return value from which cannot distinguish between failure and a
   local broadcast address.  Write a pointer to the first
   non-converted character to *endp.  */
static int
inet_aton_end (const char *cp, struct in_addr *addr, const char **endp)
{
  static const in_addr_t max[4] = { 0xffffffff, 0xffffff, 0xffff, 0xff };
  in_addr_t val;
  char c;
  union iaddr
  {
    uint8_t bytes[4];
    uint32_t word;
  } res;
  uint8_t *pp = res.bytes;
  int digit;

  int saved_errno = errno;
  __set_errno (0);

  res.word = 0;

  c = *cp;
  for (;;)
    {
      /* Collect number up to ``.''.  Values are specified as for C:
	 0x=hex, 0=octal, isdigit=decimal.  */
      if (!isdigit (c))
	goto ret_0;
      {
	char *endp;
	unsigned long ul = strtoul (cp, &endp, 0);
	if (ul == ULONG_MAX && errno == ERANGE)
	  goto ret_0;
	if (ul > 0xfffffffful)
	  goto ret_0;
	val = ul;
	digit = cp != endp;
	cp = endp;
      }
      c = *cp;
      if (c == '.')
	{
	  /* Internet format:
	     a.b.c.d
	     a.b.c	(with c treated as 16 bits)
	     a.b	(with b treated as 24 bits).  */
	  if (pp > res.bytes + 2 || val > 0xff)
	    goto ret_0;
	  *pp++ = val;
	  c = *++cp;
	}
      else
	break;
    }
  /* Check for trailing characters.  */
  if (c != '\0' && (!isascii (c) || !isspace (c)))
    goto ret_0;
  /*  Did we get a valid digit?  */
  if (!digit)
    goto ret_0;

  /* Check whether the last part is in its limits depending on the
     number of parts in total.  */
  if (val > max[pp - res.bytes])
    goto ret_0;

  if (addr != NULL)
    addr->s_addr = res.word | htonl (val);
  *endp = cp;

  __set_errno (saved_errno);
  return 1;

 ret_0:
  __set_errno (saved_errno);
  return 0;
}

int
__inet_aton_exact (const char *cp, struct in_addr *addr)
{
  struct in_addr val;
  const char *endp;
  /* Check that inet_aton_end parsed the entire string.  */
  if (inet_aton_end (cp, &val, &endp) != 0 && *endp == 0)
    {
      *addr = val;
      return 1;
    }
  else
    return 0;
}
libc_hidden_def (__inet_aton_exact)

/* inet_aton ignores trailing garbage.  */
int
__inet_aton_ignore_trailing (const char *cp, struct in_addr *addr)
{
  const char *endp;
  return  inet_aton_end (cp, addr, &endp);
}
weak_alias (__inet_aton_ignore_trailing, inet_aton)

/* ASCII IPv4 Internet address interpretation routine.  The value
   returned is in network order.  */
in_addr_t
__inet_addr (const char *cp)
{
  struct in_addr val;
  const char *endp;
  if (inet_aton_end (cp, &val, &endp))
    return val.s_addr;
  return INADDR_NONE;
}
weak_alias (__inet_addr, inet_addr)
