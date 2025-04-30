/* Check if two domain names are equal after trailing dot normalization.
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

#include <arpa/nameser.h>
#include <string.h>

/* Determines whether domain name A is the same as domain name B.
   Returns -1 on error, 0 if names differ, 1 if names are the
   same.  */
int
__libc_ns_samename (const char *a, const char *b)
{
  char ta[NS_MAXDNAME], tb[NS_MAXDNAME];

  if (__libc_ns_makecanon (a, ta, sizeof ta) < 0 ||
      __libc_ns_makecanon (b, tb, sizeof tb) < 0)
    return -1;
  if (__strcasecmp (ta, tb) == 0)
    return 1;
  else
    return 0;
}
libc_hidden_def (__libc_ns_samename)
