/* Test restarting behaviour of mbrtowc.
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

#define show(expr, nexp, wcexp) \
  n = expr;								  \
  printf (#expr " -> %zu", n);						  \
  printf (", wc = %lu", (unsigned long int) wc);			  \
  if (n != (size_t) nexp || wc != wcexp)				  \
    {									  \
      printf (", expected %zu and %lu", (size_t) nexp,			  \
	      (unsigned long int) wcexp);				  \
      result = 1;							  \
    }									  \
  putc ('\n', stdout)

static int
do_test (void)
{
  const unsigned char buf[6] = { 0x25,  0xe2, 0x82, 0xac,  0xce, 0xbb };
  mbstate_t state;
  wchar_t wc = 42;
  size_t n;
  int result = 0;
  const char *used_locale;

  setlocale (LC_CTYPE, "de_DE.UTF-8");
  /* Double check.  */
  used_locale = setlocale (LC_CTYPE, NULL);
  printf ("used locale: \"%s\"\n", used_locale);
  result = strcmp (used_locale, "de_DE.UTF-8");

  memset (&state, '\0', sizeof (state));

  show (mbrtowc (&wc, (const char *) buf + 0, 1, &state), 1, 37);
  show (mbrtowc (&wc, (const char *) buf + 1, 1, &state), -2, 37);
  show (mbrtowc (&wc, (const char *) buf + 2, 3, &state), 2, 8364);
  show (mbrtowc (&wc, (const char *) buf + 4, 1, &state), -2, 8364);
  show (mbrtowc (&wc, (const char *) buf + 5, 1, &state), 1, 955);
  show (mbrtowc (&wc, (const char *) buf + 5, 1, &state), -1, 955);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
