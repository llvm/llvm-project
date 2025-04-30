/*
 * Copyright (c) 2004 by Internet Systems Consortium, Inc. ("ISC")
 * Copyright (c) 1995,1999 by Internet Software Consortium.
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
#include <arpa/nameser.h>
#include <errno.h>
#include <string.h>

/*%
 *	Check whether a name belongs to a domain.
 *
 * Inputs:
 *\li	a - the domain whose ancestry is being verified
 *\li	b - the potential ancestor we're checking against
 *
 * Return:
 *\li	boolean - is a at or below b?
 *
 * Notes:
 *\li	Trailing dots are first removed from name and domain.
 *	Always compare complete subdomains, not only whether the
 *	domain name is the trailing string of the given name.
 *
 *\li	"host.foobar.top" lies in "foobar.top" and in "top" and in ""
 *	but NOT in "bar.top"
 */

int
ns_samedomain(const char *a, const char *b) {
	size_t la, lb;
	int diff, i, escaped;
	const char *cp;

	la = strlen(a);
	lb = strlen(b);

	/* Ignore a trailing label separator (i.e. an unescaped dot) in 'a'. */
	if (la != 0U && a[la - 1] == '.') {
		escaped = 0;
		/* Note this loop doesn't get executed if la==1. */
		for (i = la - 2; i >= 0; i--)
			if (a[i] == '\\') {
				if (escaped)
					escaped = 0;
				else
					escaped = 1;
			} else
				break;
		if (!escaped)
			la--;
	}

	/* Ignore a trailing label separator (i.e. an unescaped dot) in 'b'. */
	if (lb != 0U && b[lb - 1] == '.') {
		escaped = 0;
		/* note this loop doesn't get executed if lb==1 */
		for (i = lb - 2; i >= 0; i--)
			if (b[i] == '\\') {
				if (escaped)
					escaped = 0;
				else
					escaped = 1;
			} else
				break;
		if (!escaped)
			lb--;
	}

	/* lb == 0 means 'b' is the root domain, so 'a' must be in 'b'. */
	if (lb == 0U)
		return (1);

	/* 'b' longer than 'a' means 'a' can't be in 'b'. */
	if (lb > la)
		return (0);

	/* 'a' and 'b' being equal at this point indicates sameness. */
	if (lb == la)
		return (strncasecmp(a, b, lb) == 0);

	/* Ok, we know la > lb. */

	diff = la - lb;

	/*
	 * If 'a' is only 1 character longer than 'b', then it can't be
	 * a subdomain of 'b' (because of the need for the '.' label
	 * separator).
	 */
	if (diff < 2)
		return (0);

	/*
	 * If the character before the last 'lb' characters of 'b'
	 * isn't '.', then it can't be a match (this lets us avoid
	 * having "foobar.com" match "bar.com").
	 */
	if (a[diff - 1] != '.')
		return (0);

	/*
	 * We're not sure about that '.', however.  It could be escaped
         * and thus not a really a label separator.
	 */
	escaped = 0;
	for (i = diff - 2; i >= 0; i--)
		if (a[i] == '\\') {
			if (escaped)
				escaped = 0;
			else
				escaped = 1;
		} else
			break;
	if (escaped)
		return (0);

	/* Now compare aligned trailing substring. */
	cp = a + diff;
	return (strncasecmp(cp, b, lb) == 0);
}
libresolv_hidden_def (ns_samedomain)

/*%
 *	is "a" a subdomain of "b"?
 */
int
ns_subdomain (const char *a, const char *b)
{
  return __libc_ns_samename (a, b) != 1 && ns_samedomain (a, b);
}

/*! \file */
