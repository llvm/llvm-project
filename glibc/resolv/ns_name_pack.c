/* Compression of DNS domain names.
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
#include <string.h>
#include <shlib-compat.h>

/* Thinking in noninternationalized USASCII (per the DNS spec),
   convert this character to lower case if it's upper case.  */
static int
mklower (int ch)
{
  if (ch >= 'A' && ch <= 'Z')
    return ch - 'A' + 'a';
  return ch;
}

/* Search for the counted-label name in an array of compressed names.
   Returns the offset from MSG if found, or -1.

   DNPTRS is the pointer to the first name on the list, not the
   pointer to the start of the message.  */
static int
dn_find (const unsigned char *domain, const unsigned char *msg,
         const unsigned char **dnptrs,
         const unsigned char **lastdnptr)
{
  const unsigned char *dn, *cp, *sp;
  const unsigned char **cpp;
  unsigned int n;

  for (cpp = dnptrs; cpp < lastdnptr; cpp++)
    {
    sp = *cpp;
    /* Terminate search on: root label, compression pointer, unusable
       offset.  */
    while (*sp != 0 && (*sp & NS_CMPRSFLGS) == 0 && (sp - msg) < 0x4000)
      {
        dn = domain;
        cp = sp;
        while ((n = *cp++) != 0)
          {
            /* Check for indirection.  */
            switch (n & NS_CMPRSFLGS)
              {
              case 0:                 /* Normal case, n == len.  */
                if (n != *dn++)
                  goto next;

                for (; n > 0; n--)
                  if (mklower (*dn++) != mklower (*cp++))
                    goto next;
                /* Is next root for both?  */
                if (*dn == '\0' && *cp == '\0')
                  return sp - msg;
                if (*dn)
                  continue;
                goto next;
              case NS_CMPRSFLGS: /* Indirection.  */
                cp = msg + (((n & 0x3f) << 8) | *cp);
                break;

              default:          /* Illegal type.  */
                __set_errno (EMSGSIZE);
                return -1;
              }
          }
      next: ;
        sp += *sp + 1;
      }
    }
  __set_errno (ENOENT);
  return -1;
}

/* Packs domain name SRC into DST.  Returns size of the compressed
   name, or -1.

   DNPTRS is an array of pointers to previous compressed names.
   DNPTRS[0] is a pointer to the beginning of the message. The array
   ends with NULL.  LASTDNPTR is a pointer to the end of the array
   pointed to by 'dnptrs'.

   The list of pointers in DNPTRS is updated for labels inserted into
   the message as we compress the name.  If DNPTRS is NULL, we don't
   try to compress names. If LASTDNPTR is NULL, we don't update the
   list.  */
int
___ns_name_pack (const unsigned char *src, unsigned char *dst, int dstsiz,
                 const unsigned char **dnptrs, const unsigned char **lastdnptr)
{
  unsigned char *dstp;
  const unsigned char **cpp, **lpp, *eob, *msg;
  const unsigned char *srcp;
  int n, l, first = 1;

  srcp = src;
  dstp = dst;
  eob = dstp + dstsiz;
  lpp = cpp = NULL;
  if (dnptrs != NULL)
    {
      if ((msg = *dnptrs++) != NULL)
        {
          for (cpp = dnptrs; *cpp != NULL; cpp++)
            ;
          lpp = cpp;            /* End of list to search.  */
        }
    }
  else
    msg = NULL;

  /* Make sure the domain we are about to add is legal.  */
  l = 0;
  do
    {
      n = *srcp;
      if (n >= 64)
        {
          __set_errno (EMSGSIZE);
          return -1;
        }
      l += n + 1;
      if (l > MAXCDNAME)
        {
          __set_errno (EMSGSIZE);
          return -1;
        }
      srcp += n + 1;
    }
  while (n != 0);

  /* from here on we need to reset compression pointer array on error */
  srcp = src;
  do
    {
      /* Look to see if we can use pointers.  */
      n = *srcp;
      if (n != 0 && msg != NULL)
        {
          l = dn_find (srcp, msg, dnptrs, lpp);
          if (l >= 0)
            {
              if (eob - dstp <= 1)
                goto cleanup;
              *dstp++ = (l >> 8) | NS_CMPRSFLGS;
              *dstp++ = l % 256;
              return dstp - dst;
            }
          /* Not found, save it.  */
          if (lastdnptr != NULL && cpp < lastdnptr - 1
              && (dstp - msg) < 0x4000 && first)
            {
              *cpp++ = dstp;
              *cpp = NULL;
              first = 0;
            }
        }
      /* Copy label to buffer.  */
      if (n >= 64)
        /* Should not happen.  */
        goto cleanup;
      if (n + 1 > eob - dstp)
        goto cleanup;
      memcpy (dstp, srcp, n + 1);
      srcp += n + 1;
      dstp += n + 1;
    }
  while (n != 0);

  if (dstp > eob)
    {
    cleanup:
      if (msg != NULL)
        *lpp = NULL;
      __set_errno (EMSGSIZE);
      return -1;
    }
  return dstp - dst;
}
versioned_symbol (libc, ___ns_name_pack, ns_name_pack, GLIBC_2_34);
versioned_symbol (libc, ___ns_name_pack, __ns_name_pack, GLIBC_PRIVATE);
libc_hidden_ver (___ns_name_pack, __ns_name_pack)

#if OTHER_SHLIB_COMPAT (libresolv, GLIBC_2_9, GLIBC_2_34)
compat_symbol (libresolv, ___ns_name_pack, ns_name_pack, GLIBC_2_9);
#endif
