/* bug 17197: check that iconv doesn't emit invalid extra shift character
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

static int
do_test (void)
{
  static const char *charsets[] =
    { "IBM930", "IBM933", "IBM935", "IBM937", "IBM939" };
  static const char *expects[] =
    { "\016\x44\x4d\017", "\016\x41\x63\017", "\016\x44\x4d\017",
      "\016\x44\x4d\017", "\016\x44\x4d\017" };
  int ret = 0;

  for (int i = 0; i < sizeof (charsets) / sizeof (*charsets); i++)
    {
      const char *charset = charsets[i];
      iconv_t cd = iconv_open (charset, "UTF-8");
      if (cd == (iconv_t) -1)
	{
	  printf ("iconv_open failed (%s)\n", charset);
	  ret = 1;
	  continue;
	}

      char input[] = "\xe2\x88\x9e.";
      const char *expect1 = expects[i];
      const char expect2[] = "\x4b";
      size_t input_len = sizeof (input);
      char output[4];
      size_t inlen = input_len;
      size_t outlen = sizeof (output);
      char *inptr = input;
      char *outptr = output;
      /* First round: expect conversion to stop before ".".  */
      size_t r = iconv (cd, &inptr, &inlen, &outptr, &outlen);
      if (r != -1
	  || errno != E2BIG
	  || inlen != 2
	  || inptr != input + input_len - 2
	  || outlen != 0
	  || memcmp (output, expect1, sizeof (output)) != 0)
	{
	  printf ("wrong first conversion (%s)", charset);
	  ret = 1;
	  goto do_close;
	}

      outlen = sizeof (output);
      outptr = output;
      r = iconv (cd, &inptr, &inlen, &outptr, &outlen);
      if (r != 0
	  || inlen != 0
	  || outlen != sizeof (output) - sizeof (expect2)
	  || memcmp (output, expect2, sizeof (expect2)) != 0)
	{
	  printf ("wrong second conversion (%s)\n", charset);
	  ret = 1;
	}

    do_close:
      if (iconv_close (cd) != 0)
	{
	  printf ("iconv_close failed (%s)\n", charset);
	  ret = 1;
	  continue;
	}
    }
  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
