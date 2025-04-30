/* Test restarting behaviour of wcsrtombs.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <haible@ilog.fr>.

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
#include <string.h>
#include <wchar.h>
#include <locale.h>

#define show(expr, nexp, srcexp, bufexp) \
  {									\
    size_t res = expr;							\
    printf (#expr " -> %zu", res);					\
    dst += res;								\
    printf (", src = srcbuf+%td, dst = buf+%td",			\
	    src - srcbuf, dst - (char *) buf);				\
    if (res != nexp || src != (srcexp) || dst != (char *) (bufexp))	\
      {									\
	printf (", expected %zu and srcbuf+%td and buf+%td", (size_t) nexp, \
		(srcexp) - srcbuf, (bufexp) - (unsigned char *) buf);	\
	result = 1;							\
      }									\
    putc ('\n', stdout);						\
  }

static int
do_test (void)
{
  unsigned char buf[6] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
  const unsigned char bufcheck[6] = { 0x25, 0xe2, 0x82, 0xac, 0xce, 0xbb };
  const wchar_t srcbuf[4] = { 0x25, 0x20ac, 0x03bb, 0 };
  mbstate_t state;
  const wchar_t *src;
  char *dst;
  int result = 0;
  const char *used_locale;

  setlocale (LC_CTYPE, "de_DE.UTF-8");
  /* Double check.  */
  used_locale = setlocale (LC_CTYPE, NULL);
  printf ("used locale: \"%s\"\n", used_locale);
  result = strcmp (used_locale, "de_DE.UTF-8");

  memset (&state, '\0', sizeof (state));

  src = srcbuf;
  dst = (char *) buf;
  show (wcsrtombs (dst, &src, 1, &state), 1, srcbuf + 1, buf + 1);
  show (wcsrtombs (dst, &src, 1, &state), 0, srcbuf + 1, buf + 1);
  show (wcsrtombs (dst, &src, 4, &state), 3, srcbuf + 2, buf + 4);
  show (wcsrtombs (dst, &src, 2, &state), 2, srcbuf + 3, buf + 6);

  if (memcmp (buf, bufcheck, 6))
    {
      puts ("wrong results");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
