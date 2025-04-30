/* Test restarting behaviour of mbsnrtowcs.
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

#define show(expr, nexp, wcexp, end) \
  n = expr;								\
  printf (#expr " -> %zu", n);						\
  printf (", wc = %lu, src = buf+%td", (unsigned long int) wc,		\
	  src - (const char *) buf);					\
  if (n != (size_t) nexp || wc != wcexp || src != (const char *) (end))	\
    {									\
      printf (", expected %zu and %lu and buf+%td", (size_t) nexp,	\
	      (unsigned long int) wcexp, (end) - buf);			\
      result = 1;							\
    }									\
  putc ('\n', stdout)

static int
do_test (void)
{
  unsigned char buf[6] = { 0x25,  0xe2, 0x82, 0xac,  0xce, 0xbb };
  mbstate_t state;
  const char *src;
  wchar_t wc = 42;
  size_t n;
  int result = 0;
  const char *used_locale;

  setlocale (LC_CTYPE,"de_DE.UTF-8");
  /* Double check.  */
  used_locale = setlocale (LC_CTYPE, NULL);
  printf ("used locale: \"%s\"\n", used_locale);
  result = strcmp (used_locale, "de_DE.UTF-8");

  memset (&state, '\0', sizeof (state));

  src = (const char *) buf;
  show (mbsnrtowcs (&wc, &src, 1, 1, &state), 1, 37, buf + 1);
  show (mbsnrtowcs (&wc, &src, 3, 1, &state), 1, 8364, buf + 4);
  show (mbsnrtowcs (&wc, &src, 1, 1, &state), 0, 8364, buf + 5);
  show (mbsnrtowcs (&wc, &src, 1, 1, &state), 1, 955, buf + 6);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
