/* Syntax checking for DNS domain names.
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

#include <arpa/nameser.h>
#include <resolv.h>
#include <shlib-compat.h>

/* Return true if the string consists of printable ASCII characters
   only.  */
static bool
printable_string (const char *dn)
{
  while (true)
    {
      char ch = *dn;
      if (ch == '\0')
	return true;
      if (ch <= ' ' || ch > '~')
	return false;
      ++dn;
    }
}

/* Return true if DN points to a name consisting only of [0-9a-zA-Z_-]
   characters.  DN must be in DNS wire format, without
   compression.  */
static bool
binary_hnok (const unsigned char *dn)
{
  while (true)
    {
      size_t label_length = *dn;
      if (label_length == 0)
	break;
      ++dn;
      const unsigned char *label_end = dn + label_length;
      do
	{
	  unsigned char ch = *dn;
	  if (!(('0' <= ch && ch <= '9')
		|| ('A' <= ch && ch <= 'Z')
		|| ('a' <= ch && ch <= 'z')
		|| ch == '-' || ch == '_'))
	    return false;
	  ++dn;
	}
      while (dn < label_end);
    }
  return true;
}

/* Return true if the binary domain name has a first labels which
   starts with '-'.  */
static inline bool
binary_leading_dash (const unsigned char *dn)
{
  return dn[0] > 0 && dn[1] == '-';
}

/* Return 1 if res_hnok is a valid host name.  Labels must only
   contain [0-9a-zA-Z_-] characters, and the name must not start with
   a '-'.  The latter is to avoid confusion with program options.  */
int
___res_hnok (const char *dn)
{
  unsigned char buf[NS_MAXCDNAME];
  if (!printable_string (dn)
      || __ns_name_pton (dn, buf, sizeof (buf)) < 0
      || binary_leading_dash (buf))
    return 0;
  return binary_hnok (buf);
}
versioned_symbol (libc, ___res_hnok, res_hnok, GLIBC_2_34);
versioned_symbol (libc, ___res_hnok, __libc_res_hnok, GLIBC_PRIVATE);
libc_hidden_ver (___res_hnok, __libc_res_hnok)
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libresolv, ___res_hnok, __res_hnok, GLIBC_2_0);
#endif

/* Hostname-like (A, MX, WKS) owners can have "*" as their first label
   but must otherwise be as a host name.  */
int
___res_ownok (const char *dn)
{
  unsigned char buf[NS_MAXCDNAME];
  if (!printable_string (dn)
      || __ns_name_pton (dn, buf, sizeof (buf)) < 0
      || binary_leading_dash (buf))
    return 0;
  if (buf[0] == 1 && buf [1] == '*')
    /* Skip over the leading "*." part.  */
    return binary_hnok (buf + 2);
  else
    return binary_hnok (buf);
}
versioned_symbol (libc, ___res_ownok, res_ownok, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libresolv, ___res_ownok, __res_ownok, GLIBC_2_0);
#endif

/* SOA RNAMEs and RP RNAMEs can have any byte in their first label,
   but the rest of the name has to look like a host name.  */
int
___res_mailok (const char *dn)
{
  unsigned char buf[NS_MAXCDNAME];
  if (!printable_string (dn)
      || __ns_name_pton (dn, buf, sizeof (buf)) < 0)
    return 0;
  unsigned char label_length = buf[0];
  /* "." is a valid missing representation */
  if (label_length == 0)
    return 1;
  /* Skip over the first label.  */
  unsigned char *tail = buf + 1 + label_length;
  if (*tail == 0)
    /* More than one label is required (except for ".").  */
    return 0;
  return binary_hnok (tail);
}
versioned_symbol (libc, ___res_mailok, res_mailok, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libresolv, ___res_mailok, __res_mailok, GLIBC_2_0);
#endif

/* Return 1 if DN is a syntactically valid domain name.  Empty names
   are accepted.  */
int
___res_dnok (const char *dn)
{
  unsigned char buf[NS_MAXCDNAME];
  return printable_string (dn) && __ns_name_pton (dn, buf, sizeof (buf)) >= 0;
}
versioned_symbol (libc, ___res_dnok, res_dnok, GLIBC_2_34);
versioned_symbol (libc, ___res_dnok, __libc_res_dnok, GLIBC_PRIVATE);
libc_hidden_ver (___res_dnok, __libc_res_dnok)
#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libresolv, ___res_dnok, __res_dnok, GLIBC_2_0);
#endif
