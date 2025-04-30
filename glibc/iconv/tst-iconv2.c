/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2001.

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

#include <errno.h>
#include <iconv.h>
#include <mcheck.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int
do_test (void)
{
  char buf[3];
  const wchar_t wc[1] = L"a";
  iconv_t cd;
  char *inptr;
  size_t inlen;
  char *outptr;
  size_t outlen;
  size_t n;
  int e;
  int result = 0;

  mtrace ();

  cd = iconv_open ("UCS4", "WCHAR_T");
  if (cd == (iconv_t) -1)
    {
      printf ("cannot convert from wchar_t to UCS4: %m\n");
      exit (1);
    }

  inptr = (char *) wc;
  inlen = sizeof (wchar_t);
  outptr = buf;
  outlen = 3;

  n = iconv (cd, &inptr, &inlen, &outptr, &outlen);
  e = errno;

  if (n != (size_t) -1)
    {
      printf ("incorrect iconv() return value: %zd, expected -1\n", n);
      result = 1;
    }

  if (e != E2BIG)
    {
      printf ("incorrect error value: %s, expected %s\n",
	      strerror (e), strerror (E2BIG));
      result = 1;
    }

  if (inptr != (char *) wc)
    {
      puts ("inptr changed");
      result = 1;
    }

  if (inlen != sizeof (wchar_t))
    {
      puts ("inlen changed");
      result = 1;
    }

  if (outptr != buf)
    {
      puts ("outptr changed");
      result = 1;
    }

  if (outlen != 3)
    {
      puts ("outlen changed");
      result = 1;
    }

  iconv_close (cd);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
