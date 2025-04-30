/*
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

#include <sys/types.h>

#include <netinet/in.h>
#include <arpa/nameser.h>

#include <errno.h>
#include <resolv.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <limits.h>

# define SPRINTF(x) ((size_t)sprintf x)

/* Forward. */

static int		labellen(const u_char *);

/* Public. */

/*%
 *	Convert a network strings labels into all lowercase.
 *
 * return:
 *\li	Number of bytes written to buffer, or -1 (with errno set)
 *
 * notes:
 *\li	Enforces label and domain length limits.
 */

int
ns_name_ntol(const u_char *src, u_char *dst, size_t dstsiz)
{
	const u_char *cp;
	u_char *dn, *eom;
	u_char c;
	u_int n;
	int l;

	cp = src;
	dn = dst;
	eom = dst + dstsiz;

	if (dn >= eom) {
		__set_errno (EMSGSIZE);
		return (-1);
	}
	while ((n = *cp++) != 0) {
		if ((n & NS_CMPRSFLGS) == NS_CMPRSFLGS) {
			/* Some kind of compression pointer. */
			__set_errno (EMSGSIZE);
			return (-1);
		}
		*dn++ = n;
		if ((l = labellen(cp - 1)) < 0) {
			__set_errno (EMSGSIZE);
			return (-1);
		}
		if (dn + l >= eom) {
			__set_errno (EMSGSIZE);
			return (-1);
		}
		for ((void)NULL; l > 0; l--) {
			c = *cp++;
			if (isupper(c))
				*dn++ = tolower(c);
			else
				*dn++ = c;
		}
	}
	*dn++ = '\0';
	return (dn - dst);
}

/*%
 * Reset dnptrs so that there are no active references to pointers at or
 * after src.
 */
void
ns_name_rollback(const u_char *src, const u_char **dnptrs,
		 const u_char **lastdnptr)
{
	while (dnptrs < lastdnptr && *dnptrs != NULL) {
		if (*dnptrs >= src) {
			*dnptrs = NULL;
			break;
		}
		dnptrs++;
	}
}

/* Private. */

/* Return the length of the encoded label starting at LP, or -1 for
   compression references and extended label types.  */
static int
labellen (const unsigned char *lp)
{
  if (*lp <= 63)
    return *lp;
  return -1;
}

/*! \file */
