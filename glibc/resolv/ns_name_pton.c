/* Convert a DNS domain name from presentation to wire format.
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

/* Converts an ASCII string into an encoded domain name as per
   RFC1035.  Returns -1 if it fails, 1 if string was fully qualified,
   0 is string was not fully qualified.  Enforces label and domain
   length limits.  */
int
___ns_name_pton (const char *src, unsigned char *dst, size_t dstsiz)
{
  unsigned char *label, *bp, *eom;
  int c, n, escaped;

  escaped = 0;
  bp = dst;
  eom = dst + dstsiz;
  label = bp++;

  while ((c = *src++) != 0)
    {
      if (escaped)
        {
          if ('0' <= c && c <= '9')
            {
              n = (c - '0') * 100;
              if ((c = *src++) == 0 || c < '0' || c > '9')
                {
                  __set_errno (EMSGSIZE);
                  return -1;
                }
              n += (c - '0') * 10;
              if ((c = *src++) == 0 || c < '0' || c > '9')
                {
                  __set_errno (EMSGSIZE);
                  return -1;
                }
              n += c - '0';
              if (n > 255)
                {
                  __set_errno (EMSGSIZE);
                  return -1;
                }
              c = n;
            }
          escaped = 0;
        }
      else if (c == '\\')
        {
          escaped = 1;
          continue;
        }
      else if (c == '.')
        {
          c = (bp - label - 1);
          if ((c & NS_CMPRSFLGS) != 0) /* Label too big.  */
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          if (label >= eom)
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          *label = c;
          /* Fully qualified ? */
          if (*src == '\0')
            {
              if (c != 0)
                {
                  if (bp >= eom)
                    {
                      __set_errno (EMSGSIZE);
                      return -1;
                    }
                  *bp++ = '\0';
                }
              if ((bp - dst) > MAXCDNAME)
                {
                  __set_errno (EMSGSIZE);
                  return -1;
                }
              return 1;
            }
          if (c == 0 || *src == '.')
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          label = bp++;
          continue;
        }
      if (bp >= eom)
        {
          __set_errno (EMSGSIZE);
          return -1;
        }
      *bp++ = (unsigned char) c;
    }
  if (escaped)                  /* Trailing backslash.  */
    {
      __set_errno (EMSGSIZE);
      return -1;
    }
  c = (bp - label - 1);
  if ((c & NS_CMPRSFLGS) != 0)  /* Label too big.  */
    {
      __set_errno (EMSGSIZE);
      return -1;
    }
  if (label >= eom)
    {
      __set_errno (EMSGSIZE);
      return -1;
    }
  *label = c;
  if (c != 0)
    {
      if (bp >= eom)
        {
          __set_errno (EMSGSIZE);
          return -1;
        }
      *bp++ = 0;
    }
  if ((bp - dst) > MAXCDNAME)   /* src too big.  */
    {
      __set_errno (EMSGSIZE);
      return -1;
    }
  return 0;
}
versioned_symbol (libc, ___ns_name_pton, ns_name_pton, GLIBC_2_34);
versioned_symbol (libc, ___ns_name_pton, __ns_name_pton, GLIBC_PRIVATE);
libc_hidden_ver (___ns_name_pton, __ns_name_pton)

#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_9, GLIBC_2_34)
compat_symbol (libresolv, ___ns_name_pton, ns_name_pton, GLIBC_2_9);
#endif
