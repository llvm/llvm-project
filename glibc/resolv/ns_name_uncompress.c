/* Expand compressed domain name to presentation format.
 * Copyright (c) 2004 by Internet Systems Consortium, Inc. ("ISC")
 * Copyright (c) 1996,1999 by Internet Software Consortium.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND ISC DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS.  IN NO EVENT SHALL ISC BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <arpa/nameser.h>
#include <shlib-compat.h>

/* Expand compressed domain name to presentation format.  Returns the
   number of bytes read out of `src', or -1 (with errno set).  The
   root domain is returned as ".", not "".  */
int
___ns_name_uncompress (const unsigned char *msg, const unsigned char *eom,
                       const unsigned char *src, char *dst, size_t dstsiz)
{
  unsigned char tmp[NS_MAXCDNAME];
  int n = __ns_name_unpack (msg, eom, src, tmp, sizeof tmp);
  if (n < 0)
    return -1;
  if (__ns_name_ntop (tmp, dst, dstsiz) < 0)
    return -1;
  return n;
}
versioned_symbol (libc, ___ns_name_uncompress, ns_name_uncompress,
                  GLIBC_2_34);
versioned_symbol (libc, ___ns_name_uncompress, __ns_name_uncompress,
                  GLIBC_PRIVATE);
libc_hidden_ver (___ns_name_uncompress, __ns_name_uncompress)

#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_9, GLIBC_2_34)
compat_symbol (libresolv, ___ns_name_uncompress, ns_name_uncompress,
               GLIBC_2_9);
#endif
