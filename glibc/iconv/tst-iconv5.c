/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by GOTO Masanori <gotom@debian.or.jp>, 2004

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
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define SIZE 256		/* enough room for conversion */
#define SAMPLESTR "abc"

struct unalign
{
  char str1[1];
  char str2[SIZE];
};

struct convcode
{
  const char *tocode;
  const char *fromcode;
};

/* test builtin transformation */
static const struct convcode testcode[] = {
  {"ASCII", "ASCII"},
  {"UTF-8", "ASCII"},
  {"UCS-2BE", "ASCII"},
  {"UCS-2LE", "ASCII"},
  {"UCS-4BE", "ASCII"},
  {"UCS-4LE", "ASCII"},
};

static const int number = (int) sizeof (testcode) / sizeof (struct convcode);

static int
convert (const char *tocode, const char *fromcode, char *inbufp,
	 size_t inbytesleft, char *outbufp, size_t outbytesleft)
{
  iconv_t *ic;
  size_t outbytes = outbytesleft;
  int ret;

  ic = iconv_open (tocode, fromcode);
  if (ic == (iconv_t *) - 1)
    {
      printf ("iconv_open failed: from: %s, to: %s: %s",
	      fromcode, tocode, strerror (errno));
      return -1;
    }

  while (inbytesleft > 0)
    {
      ret = iconv (ic, &inbufp, &inbytesleft, &outbufp, &outbytes);
      if (ret == -1)
	{
	  printf ("iconv failed: from: %s, to: %s: %s",
		  fromcode, tocode, strerror (errno));
	  iconv_close (ic);
	  return -1;
	}
    }

  ret = iconv_close (ic);
  if (ret == -1)
    {
      printf ("iconv_close failed: from: %s, to: %s: %s",
	      fromcode, tocode, strerror (errno));
      return -1;
    }

  return outbytesleft - outbytes;
}


static int
test_unalign (const struct convcode *codes, const char *str, int len)
{
  struct unalign *inbufp, *outbufp;
  char *inbuf, *outbuf;
  size_t inbytesleft, outbytesleft;
  int retlen;

  /* allocating unaligned buffer for both inbuf and outbuf */
  inbufp = (struct unalign *) malloc (sizeof (struct unalign));
  if (!inbufp)
    {
      printf ("no memory available\n");
      exit (1);
    }
  inbuf = inbufp->str2;

  outbufp = (struct unalign *) malloc (sizeof (struct unalign));
  if (!outbufp)
    {
      printf ("no memory available\n");
      exit (1);
    }
  outbuf = outbufp->str2;

  /* first iconv phase */
  memcpy (inbuf, str, len);
  inbytesleft = len;
  outbytesleft = sizeof (struct unalign);
  retlen = convert (codes->tocode, codes->fromcode, inbuf, inbytesleft,
		    outbuf, outbytesleft);
  if (retlen == -1)	/* failed */
    return 1;

  /* second round trip iconv phase */
  memcpy (inbuf, outbuf, retlen);
  inbytesleft = retlen;
  outbytesleft = sizeof (struct unalign);
  retlen = convert (codes->fromcode, codes->tocode, inbuf, inbytesleft,
		    outbuf, outbytesleft);
  if (retlen == -1)	/* failed */
    return 1;

  free (inbufp);
  free (outbufp);

  return 0;
}

static int
do_test (void)
{
  int i;
  int ret = 0;

  for (i = 0; i < number; i++)
    {
      ret = test_unalign (&testcode[i], (char *) SAMPLESTR, sizeof (SAMPLESTR));
      if (ret)
	break;
      printf ("iconv: %s <-> %s: ok\n",
	      testcode[i].fromcode, testcode[i].tocode);
    }
  if (ret == 0)
    printf ("Succeeded.\n");

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
