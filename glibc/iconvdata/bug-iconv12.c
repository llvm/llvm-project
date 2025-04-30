/* bug 19727: Testing UTF conversions with UTF16 surrogates as input.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <iconv.h>
#include <byteswap.h>

static int
run_conversion (const char *from, const char *to, char *inbuf, size_t inbuflen,
		int exp_errno, int line)
{
  char outbuf[16];
  iconv_t cd;
  char *inptr;
  size_t inlen;
  char *outptr;
  size_t outlen;
  size_t n;
  int e;
  int fails = 0;

  cd = iconv_open (to, from);
  if (cd == (iconv_t) -1)
    {
      printf ("line %d: cannot convert from %s to %s: %m\n", line, from, to);
      return 1;
    }

  inptr = (char *) inbuf;
  inlen = inbuflen;
  outptr = outbuf;
  outlen = sizeof (outbuf);

  errno = 0;
  n = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  e = errno;

  if (exp_errno == 0)
    {
      if (n == (size_t) -1)
	{
	  puts ("n should be >= 0, but n == -1");
	  fails ++;
	}

      if (e != 0)
	{
	  printf ("errno should be 0: 'Success', but errno == %d: '%s'\n"
		  , e, strerror(e));
	  fails ++;
	}
    }
  else
    {
      if (n != (size_t) -1)
	{
	  printf ("n should be -1, but n == %zd\n", n);
	  fails ++;
	}

      if (e != exp_errno)
	{
	  printf ("errno should be %d: '%s', but errno == %d: '%s'\n"
		  , exp_errno, strerror (exp_errno), e, strerror (e));
	  fails ++;
	}
    }

  iconv_close (cd);

  if (fails > 0)
    {
      printf ("Errors in line %d while converting %s to %s.\n\n"
	      , line, from, to);
    }

  return fails;
}

static int
do_test (void)
{
  int fails = 0;
  char buf[4];

  /* This test runs iconv() with UTF character in range of an UTF16 surrogate.
     UTF-16 high surrogate is in range 0xD800..0xDBFF and
     UTF-16 low surrogate is in range 0xDC00..0xDFFF.
     Converting from or to UTF-xx has to report errors in those cases.
     In UTF-16, surrogate pairs with a high surrogate in front of a low
     surrogate is valid.  */

  /* Use RUN_UCS4_UTF32_INPUT to test conversion ...

     ... from INTERNAL to UTF-xx[LE|BE]:
     Converting from UCS4 to UTF-xx[LE|BE] first converts UCS4 to INTERNAL
     without checking for UTF-16 surrogate values
     and then converts from INTERNAL to UTF-xx[LE|BE].
     The latter conversion has to report an error in those cases.

     ... from UTF-32[LE|BE] to INTERNAL:
     Converting directly from UTF-32LE to UTF-8|16 is needed,
     because e.g. s390x has iconv-modules which converts directly.  */
#define RUN_UCS4_UTF32_INPUT(b0, b1, b2, b3, err, line)			\
  buf[0] = b0;								\
  buf[1] = b1;								\
  buf[2] = b2;								\
  buf[3] = b3;								\
  fails += run_conversion ("UCS4", "UTF-8", buf, 4, err, line);		\
  fails += run_conversion ("UCS4", "UTF-16LE", buf, 4, err, line);	\
  fails += run_conversion ("UCS4", "UTF-16BE", buf, 4, err, line);	\
  fails += run_conversion ("UCS4", "UTF-32LE", buf, 4, err, line);	\
  fails += run_conversion ("UCS4", "UTF-32BE", buf, 4, err, line);	\
  fails += run_conversion ("UTF-32BE", "WCHAR_T", buf, 4, err, line);	\
  fails += run_conversion ("UTF-32BE", "UTF-8", buf, 4, err, line);	\
  fails += run_conversion ("UTF-32BE", "UTF-16LE", buf, 4, err, line);	\
  fails += run_conversion ("UTF-32BE", "UTF-16BE", buf, 4, err, line);	\
  buf[0] = b3;								\
  buf[1] = b2;								\
  buf[2] = b1;								\
  buf[3] = b0;								\
  fails += run_conversion ("UTF-32LE", "WCHAR_T", buf, 4, err, line);	\
  fails += run_conversion ("UTF-32LE", "UTF-8", buf, 4, err, line);	\
  fails += run_conversion ("UTF-32LE", "UTF-16LE", buf, 4, err, line);	\
  fails += run_conversion ("UTF-32LE", "UTF-16BE", buf, 4, err, line);

  /* Use UCS4/UTF32 input of 0xD7FF.  */
  RUN_UCS4_UTF32_INPUT (0x0, 0x0, 0xD7, 0xFF, 0, __LINE__);

  /* Use UCS4/UTF32 input of 0xD800.  */
  RUN_UCS4_UTF32_INPUT (0x0, 0x0, 0xD8, 0x00, EILSEQ, __LINE__);

  /* Use UCS4/UTF32 input of 0xDBFF.  */
  RUN_UCS4_UTF32_INPUT (0x0, 0x0, 0xDB, 0xFF, EILSEQ, __LINE__);

  /* Use UCS4/UTF32 input of 0xDC00.  */
  RUN_UCS4_UTF32_INPUT (0x0, 0x0, 0xDC, 0x00, EILSEQ, __LINE__);

  /* Use UCS4/UTF32 input of 0xDFFF.  */
  RUN_UCS4_UTF32_INPUT (0x0, 0x0, 0xDF, 0xFF, EILSEQ, __LINE__);

  /* Use UCS4/UTF32 input of 0xE000.  */
  RUN_UCS4_UTF32_INPUT (0x0, 0x0, 0xE0, 0x00, 0, __LINE__);


  /* Use RUN_UTF16_INPUT to test conversion from UTF16[LE|BE] to INTERNAL.
     Converting directly from UTF-16 to UTF-8|32 is needed,
     because e.g. s390x has iconv-modules which converts directly.
     Use len == 2 or 4 to specify one or two UTF-16 characters.  */
#define RUN_UTF16_INPUT(b0, b1, b2, b3, len, err, line)			\
  buf[0] = b0;								\
  buf[1] = b1;								\
  buf[2] = b2;								\
  buf[3] = b3;								\
  fails += run_conversion ("UTF-16BE", "WCHAR_T", buf, len, err, line);	\
  fails += run_conversion ("UTF-16BE", "UTF-8", buf, len, err, line);	\
  fails += run_conversion ("UTF-16BE", "UTF-32LE", buf, len, err, line); \
  fails += run_conversion ("UTF-16BE", "UTF-32BE", buf, len, err, line); \
  buf[0] = b1;								\
  buf[1] = b0;								\
  buf[2] = b3;								\
  buf[3] = b2;								\
  fails += run_conversion ("UTF-16LE", "WCHAR_T", buf, len, err, line);	\
  fails += run_conversion ("UTF-16LE", "UTF-8", buf, len, err, line);	\
  fails += run_conversion ("UTF-16LE", "UTF-32LE", buf, len, err, line); \
  fails += run_conversion ("UTF-16LE", "UTF-32BE", buf, len, err, line);

  /* Use UTF16 input of 0xD7FF.  */
  RUN_UTF16_INPUT (0xD7, 0xFF, 0xD7, 0xFF, 4, 0, __LINE__);

  /* Use [single] UTF16 high surrogate 0xD800 [with a valid character behind].
     And check an UTF16 surrogate pair [without valid low surrogate].  */
  RUN_UTF16_INPUT (0xD8, 0x0, 0x0, 0x0, 2, EINVAL, __LINE__);
  RUN_UTF16_INPUT (0xD8, 0x0, 0xD7, 0xFF, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xD8, 0x0, 0xD8, 0x0, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xD8, 0x0, 0xE0, 0x0, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xD8, 0x0, 0xDC, 0x0, 4, 0, __LINE__);

  /* Use [single] UTF16 high surrogate 0xDBFF [with a valid character behind].
     And check an UTF16 surrogate pair [without valid low surrogate].  */
  RUN_UTF16_INPUT (0xDB, 0xFF, 0x0, 0x0, 2, EINVAL, __LINE__);
  RUN_UTF16_INPUT (0xDB, 0xFF, 0xD7, 0xFF, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDB, 0xFF, 0xDB, 0xFF, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDB, 0xFF, 0xE0, 0x0, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDB, 0xFF, 0xDF, 0xFF, 4, 0, __LINE__);

  /* Use single UTF16 low surrogate 0xDC00 [with a valid character behind].
     And check an UTF16 surrogate pair [without valid high surrogate].   */
  RUN_UTF16_INPUT (0xDC, 0x0, 0x0, 0x0, 2, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDC, 0x0, 0xD7, 0xFF, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xD8, 0x0, 0xDC, 0x0, 4, 0, __LINE__);
  RUN_UTF16_INPUT (0xD7, 0xFF, 0xDC, 0x0, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDC, 0x0, 0xDC, 0x0, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xE0, 0x0, 0xDC, 0x0, 4, EILSEQ, __LINE__);

  /* Use single UTF16 low surrogate 0xDFFF [with a valid character behind].
     And check an UTF16 surrogate pair [without valid high surrogate].   */
  RUN_UTF16_INPUT (0xDF, 0xFF, 0x0, 0x0, 2, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDF, 0xFF, 0xD7, 0xFF, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDB, 0xFF, 0xDF, 0xFF, 4, 0, __LINE__);
  RUN_UTF16_INPUT (0xD7, 0xFF, 0xDF, 0xFF, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xDF, 0xFF, 0xDF, 0xFF, 4, EILSEQ, __LINE__);
  RUN_UTF16_INPUT (0xE0, 0x0, 0xDF, 0xFF, 4, EILSEQ, __LINE__);

  /* Use UCS4/UTF32 input of 0xE000.  */
  RUN_UTF16_INPUT (0xE0, 0x0, 0xE0, 0x0, 4, 0, __LINE__);


  /* Use RUN_UTF8_3BYTE_INPUT to test conversion from UTF-8 to INTERNAL.
     Converting directly from UTF-8 to UTF-16|32 is needed,
     because e.g. s390x has iconv-modules which converts directly.  */
#define RUN_UTF8_3BYTE_INPUT(b0, b1, b2, err, line)			\
  buf[0] = b0;								\
  buf[1] = b1;								\
  buf[2] = b2;								\
  fails += run_conversion ("UTF-8", "WCHAR_T", buf, 3, err, line);	\
  fails += run_conversion ("UTF-8", "UTF-16LE", buf, 3, err, line);	\
  fails += run_conversion ("UTF-8", "UTF-16BE", buf, 3, err, line);	\
  fails += run_conversion ("UTF-8", "UTF-32LE", buf, 3, err, line);	\
  fails += run_conversion ("UTF-8", "UTF-32BE", buf, 3, err, line);

  /* Use UTF-8 input of 0xD7FF.  */
  RUN_UTF8_3BYTE_INPUT (0xED, 0x9F, 0xBF, 0, __LINE__);

  /* Use UTF-8 input of 0xD800.  */
  RUN_UTF8_3BYTE_INPUT (0xED, 0xA0, 0x80, EILSEQ, __LINE__);

  /* Use UTF-8 input of 0xDBFF.  */
  RUN_UTF8_3BYTE_INPUT (0xED, 0xAF, 0xBF, EILSEQ, __LINE__);

  /* Use UTF-8 input of 0xDC00.  */
  RUN_UTF8_3BYTE_INPUT (0xED, 0xB0, 0x80, EILSEQ, __LINE__);

  /* Use UTF-8 input of 0xDFFF.  */
  RUN_UTF8_3BYTE_INPUT (0xED, 0xBF, 0xBF, EILSEQ, __LINE__);

  /* Use UTF-8 input of 0xF000.  */
  RUN_UTF8_3BYTE_INPUT (0xEF, 0x80, 0x80, 0, __LINE__);

  return fails > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
