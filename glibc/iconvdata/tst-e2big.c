/* Test for a tricky E2BIG situation.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <bruno@clisp.org>, 2002.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <alloca.h>
#include <errno.h>
#include <iconv.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* In EUC-JISX0213 and TSCII, a single input character can convert to
   a sequence of two or more Unicode characters.  When the output buffer
   has room for less Unicode characters than would be produced with an
   unconstrained output buffer, the conversion must give errno = E2BIG.  */

void
test (const char *encoding, char *inbuf, size_t inbufsize, size_t outbufsize)
{
  char *outbuf = alloca (outbufsize);
  iconv_t cd;
  char *inptr;
  size_t inlen;
  char *outptr;
  size_t outlen;
  int result;
  bool empty_input;
  bool empty_output;

  cd = iconv_open ("UTF-8", encoding);
  if (cd == (iconv_t) -1)
    {
      fprintf (stderr, "cannot convert from %s\n", encoding);
      exit (1);
    }

  inptr = inbuf;
  inlen = inbufsize;
  outptr = outbuf;
  outlen = outbufsize;

  result = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  if (!(result == -1 && errno == E2BIG))
    {
      fprintf (stderr, "%s: wrong iconv result: %d/%d (%m)\n",
	       encoding, result, errno);
      exit (1);
    }
  empty_input = (inptr == inbuf && inlen == inbufsize);
  empty_output = (outptr == outbuf && outlen == outbufsize);

  if (!empty_input && empty_output)
    {
      fprintf (stderr, "%s: ate %td input bytes\n", encoding, inptr - inbuf);
      exit (1);
    }
  if (empty_input && !empty_output)
    {
      fprintf (stderr, "%s: produced %td output bytes\n",
	       encoding, outptr - outbuf);
      exit (1);
    }

  iconv_close (cd);
}

void
test_euc_jisx0213 (void)
{
  char inbuf[2] = { 0xa4, 0xf7 };
  test ("EUC-JISX0213", inbuf, sizeof (inbuf), 3);
}

void
test_tscii (void)
{
  char inbuf[1] = { 0x82 };
  test ("TSCII", inbuf, sizeof (inbuf), 3);
  test ("TSCII", inbuf, sizeof (inbuf), 6);
  test ("TSCII", inbuf, sizeof (inbuf), 9);
}

static int
do_test (void)
{
  test_euc_jisx0213 ();
  test_tscii ();
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
