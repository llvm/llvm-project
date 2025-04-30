/* Convert DNS domain names from network format to textual presentation format.
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
#include <stdbool.h>

/* Thinking in noninternationalized US-ASCII (per the DNS spec), is
   this character special ("in need of quoting")?  */
static inline bool
special (int ch)
{
  switch (ch)
    {
    case '"':
    case '.':
    case ';':
    case '\\':
    case '(':
    case ')':
      /* Special modifiers in zone files.  */
    case '@':
    case '$':
      return true;
    default:
      return false;
    }
}

/* Thinking in noninternationalized US-ASCII (per the DNS spec), is
   this character visible and not a space when printed?  */
static inline bool
printable (int ch)
{
  return ch > 0x20 && ch < 0x7f;
}

/* Converts an uncompressed, encoded domain name to printable ASCII as
   per RFC1035.  Returns the number of bytes written to buffer, or -1
   (with errno set).  The root is returned as "."  All other domains
   are returned in non absolute form.  */
int
___ns_name_ntop (const unsigned char *src, char *dst, size_t dstsiz)
{
  const unsigned char *cp;
  char *dn, *eom;
  unsigned char c;
  int l;

  cp = src;
  dn = dst;
  eom = dst + dstsiz;

  while ((l = *cp++) != 0)
    {
      if (l >= 64)
        {
          /* Some kind of compression pointer.  */
          __set_errno (EMSGSIZE);
          return -1;
        }
      if (dn != dst)
        {
          if (dn >= eom)
            {
              __set_errno (EMSGSIZE);
              return -1;
            }
          *dn++ = '.';
        }
      for (; l > 0; l--)
        {
          c = *cp++;
          if (special (c))
            {
              if (eom - dn < 2)
                {
                  __set_errno (EMSGSIZE);
                  return -1;
                }
              *dn++ = '\\';
              *dn++ = c;
            }
          else if (!printable (c))
            {
              if (eom - dn < 4)
                {
                  __set_errno (EMSGSIZE);
                  return -1;
                }
              *dn++ = '\\';
              *dn++ = '0' + (c / 100);
              *dn++ = '0' + ((c % 100) / 10);
              *dn++ = '0' + (c % 10);
            }
          else
            {
              if (eom - dn < 2)
                {
                  __set_errno (EMSGSIZE);
                  return -1;
                }
              *dn++ = c;
            }
        }
    }
  if (dn == dst)
    {
      if (dn >= eom)
        {
          __set_errno (EMSGSIZE);
          return -1;
        }
      *dn++ = '.';
    }
  if (dn >= eom)
    {
      __set_errno (EMSGSIZE);
      return -1;
    }
  *dn++ = '\0';
  return dn - dst;
}
versioned_symbol (libc, ___ns_name_ntop, ns_name_ntop, GLIBC_2_34);
versioned_symbol (libc, ___ns_name_ntop, __ns_name_ntop, GLIBC_PRIVATE);
libc_hidden_ver (___ns_name_ntop, __ns_name_ntop)

#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_9, GLIBC_2_34)
compat_symbol (libresolv, ___ns_name_ntop, ns_name_ntop, GLIBC_2_9);
#endif
