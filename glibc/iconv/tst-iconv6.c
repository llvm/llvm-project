/* Testing ucs4le_internal_loop() in gconv_simple.c.
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
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <iconv.h>
#include <byteswap.h>
#include <endian.h>

static int
do_test (void)
{
  iconv_t cd;
  char *inptr;
  size_t inlen;
  char *outptr;
  size_t outlen;
  size_t n;
  int e;
  int result = 0;

#if __BYTE_ORDER == __BIG_ENDIAN
  /* On big-endian machines, ucs4le_internal_loop() swaps the bytes before
     error checking. Thus the input values has to be swapped.  */
# define VALUE(val) bswap_32 (val)
#else
# define VALUE(val) val
#endif
  uint32_t inbuf[3] = { VALUE (0x41), VALUE (0x80000000), VALUE (0x42) };
  uint32_t outbuf[3] = { 0, 0, 0 };

  cd = iconv_open ("WCHAR_T", "UCS-4LE");
  if (cd == (iconv_t) -1)
    {
      printf ("cannot convert from UCS4LE to wchar_t: %m\n");
      return 1;
    }

  inptr = (char *) inbuf;
  inlen = sizeof (inbuf);
  outptr = (char *) outbuf;
  outlen = sizeof (outbuf);

  n = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  e = errno;

  if (n != (size_t) -1)
    {
      printf ("incorrect iconv() return value: %zd, expected -1\n", n);
      result = 1;
    }

  if (e != EILSEQ)
    {
      printf ("incorrect error value: %s, expected %s\n",
	      strerror (e), strerror (EILSEQ));
      result = 1;
    }

  if (inptr != (char *) &inbuf[1])
    {
      printf ("inptr=0x%p does not point to invalid character! Expected=0x%p\n"
	      , inptr, &inbuf[1]);
      result = 1;
    }

  if (inlen != sizeof (inbuf) - sizeof (uint32_t))
    {
      printf ("inlen=%zd != %zd\n"
	      , inlen, sizeof (inbuf) - sizeof (uint32_t));
      result = 1;
    }

  if (outptr != (char *) &outbuf[1])
    {
      printf ("outptr=0x%p does not point to invalid character in inbuf! "
	      "Expected=0x%p\n"
	      , outptr, &outbuf[1]);
      result = 1;
    }

  if (outlen != sizeof (inbuf) - sizeof (uint32_t))
    {
      printf ("outlen=%zd != %zd\n"
	      , outlen, sizeof (outbuf) - sizeof (uint32_t));
      result = 1;
    }

  if (outbuf[0] != 0x41 || outbuf[1] != 0 || outbuf[2] != 0)
    {
      puts ("Characters conversion is incorrect!");
      result = 1;
    }

  iconv_close (cd);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
