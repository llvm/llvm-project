/* Skip over a (potentially compressed) domain name in wire format.
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

/* Advances *PTRPTR to skip over the compressed name it points at.
   Returns 0 on success, -1 (with errno set) on failure.  */
int
___ns_name_skip (const unsigned char **ptrptr, const unsigned char *eom)
{
  const unsigned char *cp;
  unsigned int n;

  cp = *ptrptr;
  while (cp < eom)
    {
      n = *cp++;
      if (n == 0)
        {
          /* End of domain name without indirection.  */
          *ptrptr = cp;
          return 0;
        }

      /* Check for indirection.  */
      switch (n & NS_CMPRSFLGS)
        {
        case 0:                 /* Normal case, n == len.  */
          if (eom - cp < n)
            goto malformed;
          cp += n;
          break;
        case NS_CMPRSFLGS:      /* Indirection.  */
          if (cp == eom)
            /* No room for second indirection byte.  */
            goto malformed;
          *ptrptr = cp + 1;
          return 0;
        default:                /* Illegal type.  */
          goto malformed;
        }
    }

 malformed:
  __set_errno (EMSGSIZE);
  return -1;
}
versioned_symbol (libc, ___ns_name_skip, ns_name_skip, GLIBC_2_34);
versioned_symbol (libc, ___ns_name_skip, __ns_name_skip, GLIBC_PRIVATE);
libc_hidden_ver (___ns_name_skip, __ns_name_skip)

#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_9, GLIBC_2_34)
compat_symbol (libresolv, ___ns_name_skip, ns_name_skip, GLIBC_2_9);
#endif
