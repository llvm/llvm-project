/* Assertion in ISO-2022-JP-3 due to two-character sequence (bug 27256).
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <iconv.h>
#include <string.h>
#include <errno.h>
#include <support/check.h>

/* Use an escape sequence to return to the initial state.  */
static void
with_escape_sequence (void)
{
  iconv_t c = iconv_open ("UTF-8", "ISO-2022-JP-3");
  TEST_VERIFY_EXIT (c != (iconv_t) -1);

  char in[] = "\e$(O+D\e(B";
  char *inbuf = in;
  size_t inleft = strlen (in);
  char out[3];                  /* Space for one output character.  */
  char *outbuf;
  size_t outleft;

  outbuf = out;
  outleft = sizeof (out);
  TEST_COMPARE (iconv (c, &inbuf, &inleft, &outbuf, &outleft), (size_t) -1);
  TEST_COMPARE (errno, E2BIG);
  TEST_COMPARE (inleft, 3);
  TEST_COMPARE (inbuf - in, strlen (in) - 3);
  TEST_COMPARE (outleft, sizeof (out) - 2);
  TEST_COMPARE (outbuf - out, 2);
  TEST_COMPARE (out[0] & 0xff, 0xc3);
  TEST_COMPARE (out[1] & 0xff, 0xa6);

  /* Return to the initial shift state, producing the pending
     character.  */
  outbuf = out;
  outleft = sizeof (out);
  TEST_COMPARE (iconv (c, &inbuf, &inleft, &outbuf, &outleft), 0);
  TEST_COMPARE (inleft, 0);
  TEST_COMPARE (inbuf - in, strlen (in));
  TEST_COMPARE (outleft, sizeof (out) - 2);
  TEST_COMPARE (outbuf - out, 2);
  TEST_COMPARE (out[0] & 0xff, 0xcc);
  TEST_COMPARE (out[1] & 0xff, 0x80);

  /* Nothing should be flushed the second time.  */
  outbuf = out;
  outleft = sizeof (out);
  TEST_COMPARE (iconv (c, NULL, 0, &outbuf, &outleft), 0);
  TEST_COMPARE (outleft, sizeof (out));
  TEST_COMPARE (outbuf - out, 0);
  TEST_COMPARE (out[0] & 0xff, 0xcc);
  TEST_COMPARE (out[1] & 0xff, 0x80);

  TEST_COMPARE (iconv_close (c), 0);
}

/* Use an explicit flush to return to the initial state.  */
static void
with_flush (void)
{
  iconv_t c = iconv_open ("UTF-8", "ISO-2022-JP-3");
  TEST_VERIFY_EXIT (c != (iconv_t) -1);

  char in[] = "\e$(O+D";
  char *inbuf = in;
  size_t inleft = strlen (in);
  char out[3];                  /* Space for one output character.  */
  char *outbuf;
  size_t outleft;

  outbuf = out;
  outleft = sizeof (out);
  TEST_COMPARE (iconv (c, &inbuf, &inleft, &outbuf, &outleft), (size_t) -1);
  TEST_COMPARE (errno, E2BIG);
  TEST_COMPARE (inleft, 0);
  TEST_COMPARE (inbuf - in, strlen (in));
  TEST_COMPARE (outleft, sizeof (out) - 2);
  TEST_COMPARE (outbuf - out, 2);
  TEST_COMPARE (out[0] & 0xff, 0xc3);
  TEST_COMPARE (out[1] & 0xff, 0xa6);

  /* Flush the pending character.  */
  outbuf = out;
  outleft = sizeof (out);
  TEST_COMPARE (iconv (c, NULL, 0, &outbuf, &outleft), 0);
  TEST_COMPARE (outleft, sizeof (out) - 2);
  TEST_COMPARE (outbuf - out, 2);
  TEST_COMPARE (out[0] & 0xff, 0xcc);
  TEST_COMPARE (out[1] & 0xff, 0x80);

  /* Nothing should be flushed the second time.  */
  outbuf = out;
  outleft = sizeof (out);
  TEST_COMPARE (iconv (c, NULL, 0, &outbuf, &outleft), 0);
  TEST_COMPARE (outleft, sizeof (out));
  TEST_COMPARE (outbuf - out, 0);
  TEST_COMPARE (out[0] & 0xff, 0xcc);
  TEST_COMPARE (out[1] & 0xff, 0x80);

  TEST_COMPARE (iconv_close (c), 0);
}

static int
do_test (void)
{
  with_escape_sequence ();
  with_flush ();
  return 0;
}

#include <support/test-driver.c>
