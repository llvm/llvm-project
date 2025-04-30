/* Add missing "." to domain names.
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
#include <errno.h>
#include <string.h>

/* Make a canonical copy of domain name SRC in DST.  Behavior:
      foo -> foo.
      foo. -> foo.
      foo.. -> foo.
      foo\. -> foo\..
      foo\\. -> foo\\.  */
int
__libc_ns_makecanon (const char *src, char *dst, size_t dstsize)
{
  size_t n = strlen (src);

  if (n + sizeof "." > dstsize) /* sizeof == 2.  */
    {
      __set_errno (EMSGSIZE);
      return -1;
    }
  strcpy (dst, src);
  while (n >= 1U && dst[n - 1] == '.')   /* Ends in ".".  */
    if (n >= 2U && dst[n - 2] == '\\' && /* Ends in "\.".  */
        (n < 3U || dst[n - 3] != '\\'))  /* But not "\\.".  */
      break;
    else
      dst[--n] = '\0';
  dst[n++] = '.';
  dst[n] = '\0';
  return 0;
}
libc_hidden_def (__libc_ns_makecanon)
