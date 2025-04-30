/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <haible@clisp.cons.org>, 2000.

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

/* Create a table from Unicode to CHARSET.
   This is a good test for CHARSET's iconv() module, in particular the
   TO_LOOP BODY macro.  */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iconv.h>
#include <errno.h>

int
main (int argc, char *argv[])
{
  const char *charset;
  iconv_t cd;
  int bmp_only;

  if (argc != 2)
    {
      fprintf (stderr, "Usage: tst-table-to charset\n");
      return 1;
    }
  charset = argv[1];

  cd = iconv_open (charset, "UTF-8");
  if (cd == (iconv_t)(-1))
    {
      perror ("iconv_open");
      return 1;
    }

  /* When testing UTF-8 or GB18030, stop at 0x10000, otherwise the output
     file gets too big.  */
  bmp_only = (strcmp (charset, "UTF-8") == 0
	      || strcmp (charset, "GB18030") == 0);

  {
    unsigned int i;
    unsigned char buf[10];

    for (i = 0; i < (bmp_only ? 0x10000 : 0x30000); i++)
      {
	unsigned char in[6];
	unsigned int incount =
	  (i < 0x80 ? (in[0] = i, 1)
	   : i < 0x800 ? (in[0] = 0xc0 | (i >> 6),
			  in[1] = 0x80 | (i & 0x3f), 2)
	   : i < 0x10000 ? (in[0] = 0xe0 | (i >> 12),
			    in[1] = 0x80 | ((i >> 6) & 0x3f),
			    in[2] = 0x80 | (i & 0x3f), 3)
	   : /* i < 0x200000 */ (in[0] = 0xf0 | (i >> 18),
				 in[1] = 0x80 | ((i >> 12) & 0x3f),
				 in[2] = 0x80 | ((i >> 6) & 0x3f),
				 in[3] = 0x80 | (i & 0x3f), 4));
	const char *inbuf = (const char *) in;
	size_t inbytesleft = incount;
	char *outbuf = (char *) buf;
	size_t outbytesleft = sizeof (buf);
	size_t result;
	size_t result2 = 0;

	iconv (cd, NULL, NULL, NULL, NULL);
	result = iconv (cd,
			(char **) &inbuf, &inbytesleft,
			&outbuf, &outbytesleft);
	if (result != (size_t)(-1))
	  result2 = iconv (cd, NULL, NULL, &outbuf, &outbytesleft);

	if (result == (size_t)(-1) || result2 == (size_t)(-1))
	  {
	    if (errno != EILSEQ)
	      {
		int saved_errno = errno;
		fprintf (stderr, "0x%02X: iconv error: ", i);
		errno = saved_errno;
		perror ("");
		return 1;
	      }
	  }
	else if (result == 0) /* ignore conversions with transliteration */
	  {
	    unsigned int j, jmax;
	    if (inbytesleft != 0 || outbytesleft == sizeof (buf))
	      {
		fprintf (stderr, "0x%02X: inbytes = %ld, outbytes = %ld\n", i,
			 (long) (incount - inbytesleft),
			 (long) (sizeof (buf) - outbytesleft));
		return 1;
	      }
	    jmax = sizeof (buf) - outbytesleft;
	    printf ("0x");
	    for (j = 0; j < jmax; j++)
	      printf ("%02X", buf[j]);
	    printf ("\t0x%04X\n", i);
	  }
      }
  }

  if (iconv_close (cd) < 0)
    {
      perror ("iconv_close");
      return 1;
    }

  if (ferror (stdin) || fflush (stdout) || ferror (stdout))
    {
      fprintf (stderr, "I/O error\n");
      return 1;
    }

  return 0;
}
