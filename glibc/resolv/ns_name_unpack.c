/* De-compressing DNS domain names into binary-encoded uncompressed name.
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
#include <errno.h>
#include <shlib-compat.h>
#include <stddef.h>
#include <string.h>

/* Unpack a domain name from a message, source may be compressed.
   Returns -1 if it fails, or consumed octets if it succeeds.  */
int
___ns_name_unpack (const unsigned char *msg, const unsigned char *eom,
                   const unsigned char *src, unsigned char *dst, size_t dstsiz)
{
  const unsigned char *srcp, *dstlim;
  unsigned char *dstp;
  int n, len, checked;

  len = -1;
  checked = 0;
  dstp = dst;
  srcp = src;
  dstlim = dst + dstsiz;
  if (srcp < msg || srcp >= eom)
    {
      __set_errno (EMSGSIZE);
      return -1;
    }
  /* Fetch next label in domain name.  */
  while ((n = *srcp++) != 0)
    {
      /* Check for indirection.  */
      switch (n & NS_CMPRSFLGS)
        {
        case 0:
          /* Limit checks.  */
          if (n >= 64)
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          /* NB: n + 1 and >= to cover the *dstp = '\0' assignment
             below.  */
          if (n + 1 >= dstlim - dstp || n >= eom - srcp)
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          checked += n + 1;
          *dstp++ = n;
          memcpy (dstp, srcp, n);
          dstp += n;
          srcp += n;
          break;

        case NS_CMPRSFLGS:
          if (srcp >= eom)
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          if (len < 0)
            len = srcp - src + 1;
          {
            int target = ((n & 0x3f) << 8) | *srcp;
            if (target >= eom - msg)
              {
              /* Out of range.  */
                __set_errno (EMSGSIZE);
                return -1;
            }
            srcp = msg + target;
          }
          checked += 2;
          /* Check for loops in the compressed name; if we've looked
             at the whole message, there must be a loop.  */
          if (checked >= eom - msg)
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          break;

        default:
          __set_errno (EMSGSIZE);
          return -1;
        }
    }
  *dstp = '\0';
  if (len < 0)
    len = srcp - src;
  return len;
}
versioned_symbol (libc, ___ns_name_unpack, ns_name_unpack, GLIBC_2_34);
versioned_symbol (libc, ___ns_name_unpack, __ns_name_unpack, GLIBC_PRIVATE);
libc_hidden_ver (___ns_name_unpack, __ns_name_unpack)

#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_9, GLIBC_2_34)
compat_symbol (libresolv, ___ns_name_unpack, ns_name_unpack, GLIBC_2_9);
#endif
