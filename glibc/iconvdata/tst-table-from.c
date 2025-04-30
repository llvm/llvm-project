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

/* Create a table from CHARSET to Unicode.
   This is a good test for CHARSET's iconv() module, in particular the
   FROM_LOOP BODY macro.  */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iconv.h>
#include <errno.h>

/* If nonzero, ignore conversions outside Unicode plane 0.  */
static int bmp_only;

/* Converts a byte buffer to a hexadecimal string.  */
static const char*
hexbuf (unsigned char buf[], unsigned int buflen)
{
  static char msg[50];

  switch (buflen)
    {
    case 1:
      sprintf (msg, "0x%02X", buf[0]);
      break;
    case 2:
      sprintf (msg, "0x%02X%02X", buf[0], buf[1]);
      break;
    case 3:
      sprintf (msg, "0x%02X%02X%02X", buf[0], buf[1], buf[2]);
      break;
    case 4:
      sprintf (msg, "0x%02X%02X%02X%02X", buf[0], buf[1], buf[2], buf[3]);
      break;
    default:
      abort ();
    }
  return msg;
}

/* Attempts to convert a byte buffer BUF (BUFLEN bytes) to OUT (12 bytes)
   using the conversion descriptor CD.  Returns the number of written bytes,
   or 0 if ambiguous, or -1 if invalid.  */
static int
try (iconv_t cd, unsigned char buf[], unsigned int buflen, unsigned char *out)
{
  const char *inbuf = (const char *) buf;
  size_t inbytesleft = buflen;
  char *outbuf = (char *) out;
  size_t outbytesleft = 12;
  size_t result;

  iconv (cd, NULL, NULL, NULL, NULL);
  result = iconv (cd, (char **) &inbuf, &inbytesleft, &outbuf, &outbytesleft);
  if (result != (size_t)(-1))
    result = iconv (cd, NULL, NULL, &outbuf, &outbytesleft);

  if (result == (size_t)(-1))
    {
      if (errno == EILSEQ)
	{
	  return -1;
	}
      else if (errno == EINVAL)
	{
	  return 0;
	}
      else
	{
	  int saved_errno = errno;
	  fprintf (stderr, "%s: iconv error: ", hexbuf (buf, buflen));
	  errno = saved_errno;
	  perror ("");
	  exit (1);
	}
    }
  else
    {
      if (inbytesleft != 0)
	{
	  fprintf (stderr, "%s: inbytes = %ld, outbytes = %ld\n",
		   hexbuf (buf, buflen),
		   (long) (buflen - inbytesleft),
		   (long) (12 - outbytesleft));
	  exit (1);
	}
      return 12 - outbytesleft;
    }
}

/* Returns the out[] buffer as a Unicode value, formatted as 0x%04X.  */
static const char *
utf8_decode (const unsigned char *out, unsigned int outlen)
{
  static char hexbuf[84];
  char *p = hexbuf;

  while (outlen > 0)
    {
      if (p > hexbuf)
	*p++ = ' ';

      if (out[0] < 0x80)
	{
	  sprintf (p, "0x%04X", out[0]);
	  out += 1; outlen -= 1;
	}
      else if (out[0] >= 0xc0 && out[0] < 0xe0 && outlen >= 2)
	{
	  sprintf (p, "0x%04X", ((out[0] & 0x1f) << 6) + (out[1] & 0x3f));
	  out += 2; outlen -= 2;
	}
      else if (out[0] >= 0xe0 && out[0] < 0xf0 && outlen >= 3)
	{
	  sprintf (p, "0x%04X", ((out[0] & 0x0f) << 12)
				+ ((out[1] & 0x3f) << 6) + (out[2] & 0x3f));
	  out += 3; outlen -= 3;
	}
      else if (out[0] >= 0xf0 && out[0] < 0xf8 && outlen >= 4)
	{
	  sprintf (p, "0x%04X", ((out[0] & 0x07) << 18)
				+ ((out[1] & 0x3f) << 12)
				+ ((out[2] & 0x3f) << 6) + (out[3] & 0x3f));
	  out += 4; outlen -= 4;
	}
      else if (out[0] >= 0xf8 && out[0] < 0xfc && outlen >= 5)
	{
	  sprintf (p, "0x%04X", ((out[0] & 0x03) << 24)
				+ ((out[1] & 0x3f) << 18)
				+ ((out[2] & 0x3f) << 12)
				+ ((out[3] & 0x3f) << 6) + (out[4] & 0x3f));
	  out += 5; outlen -= 5;
	}
      else if (out[0] >= 0xfc && out[0] < 0xfe && outlen >= 6)
	{
	  sprintf (p, "0x%04X", ((out[0] & 0x01) << 30)
				+ ((out[1] & 0x3f) << 24)
				+ ((out[2] & 0x3f) << 18)
				+ ((out[3] & 0x3f) << 12)
				+ ((out[4] & 0x3f) << 6) + (out[5] & 0x3f));
	  out += 6; outlen -= 6;
	}
      else
	{
	  sprintf (p, "0x????");
	  out += 1; outlen -= 1;
	}

      if (bmp_only && strlen (p) > 6)
	/* Ignore conversions outside Unicode plane 0.  */
	return NULL;

      p += strlen (p);
    }

  return hexbuf;
}

int
main (int argc, char *argv[])
{
  const char *charset;
  iconv_t cd;
  int search_depth;

  if (argc != 2)
    {
      fprintf (stderr, "Usage: tst-table-from charset\n");
      exit (1);
    }
  charset = argv[1];

  cd = iconv_open ("UTF-8", charset);
  if (cd == (iconv_t)(-1))
    {
      perror ("iconv_open");
      exit (1);
    }

  /* When testing UTF-8 or GB18030, stop at 0x10000, otherwise the output
     file gets too big.  */
  bmp_only = (strcmp (charset, "UTF-8") == 0
	      || strcmp (charset, "GB18030") == 0);
  search_depth = (strcmp (charset, "UTF-8") == 0 ? 3 : 4);

  {
    unsigned char out[12];
    unsigned char buf[4];
    unsigned int i0, i1, i2, i3;
    int result;

    for (i0 = 0; i0 < 0x100; i0++)
      {
	buf[0] = i0;
	result = try (cd, buf, 1, out);
	if (result < 0)
	  {
	  }
	else if (result > 0)
	  {
	    const char *unicode = utf8_decode (out, result);
	    if (unicode != NULL)
	      printf ("0x%02X\t%s\n", i0, unicode);
	  }
	else
	  {
	    for (i1 = 0; i1 < 0x100; i1++)
	      {
		buf[1] = i1;
		result = try (cd, buf, 2, out);
		if (result < 0)
		  {
		  }
		else if (result > 0)
		  {
		    const char *unicode = utf8_decode (out, result);
		    if (unicode != NULL)
		      printf ("0x%02X%02X\t%s\n", i0, i1, unicode);
		  }
		else
		  {
		    for (i2 = 0; i2 < 0x100; i2++)
		      {
			buf[2] = i2;
			result = try (cd, buf, 3, out);
			if (result < 0)
			  {
			  }
			else if (result > 0)
			  {
			    const char *unicode = utf8_decode (out, result);
			    if (unicode != NULL)
			      printf ("0x%02X%02X%02X\t%s\n",
				      i0, i1, i2, unicode);
			  }
			else if (search_depth > 3)
			  {
			    for (i3 = 0; i3 < 0x100; i3++)
			      {
				buf[3] = i3;
				result = try (cd, buf, 4, out);
				if (result < 0)
				  {
				  }
				else if (result > 0)
				  {
				    const char *unicode =
				      utf8_decode (out, result);
				    if (unicode != NULL)
				      printf ("0x%02X%02X%02X%02X\t%s\n",
					      i0, i1, i2, i3, unicode);
				  }
				else
				  {
				    fprintf (stderr,
					     "%s: incomplete byte sequence\n",
					     hexbuf (buf, 4));
				    exit (1);
				  }
			      }
			  }
		      }
		  }
	      }
	  }
      }
  }

  if (iconv_close (cd) < 0)
    {
      perror ("iconv_close");
      exit (1);
    }

  if (ferror (stdin) || fflush (stdout) || ferror (stdout))
    {
      fprintf (stderr, "I/O error\n");
      exit (1);
    }

  return 0;
}
