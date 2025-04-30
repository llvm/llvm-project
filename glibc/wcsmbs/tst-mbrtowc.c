/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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

/* We always want assert to be fully defined.  */
#undef NDEBUG
#include <assert.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>


static int check_ascii (const char *locname);

/* UTF-8 single byte feeding test for mbrtowc(),
   contributed by Markus Kuhn <mkuhn@acm.org>.  */
static int
utf8_test_1 (void)
{
  wchar_t wc;
  mbstate_t s;

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, "\xE2", 1, &s) == (size_t) -2);	/* 1st byte processed */
  assert (mbrtowc (&wc, "\x89", 1, &s) == (size_t) -2);	/* 2nd byte processed */
  assert (wc == 42);		/* no value has not been stored into &wc yet */
  assert (mbrtowc (&wc, "\xA0", 1, &s) == 1);	/* 3nd byte processed */
  assert (wc == 0x2260);	/* E2 89 A0 = U+2260 (not equal) decoded correctly */
  assert (mbrtowc (&wc, "", 1, &s) == 0);	/* test final byte processing */
  assert (wc == 0);		/* test final byte decoding */

  /* The following test is by Al Viro <aviro@redhat.com>.  */
  const char str[] = "\xe0\xa0\x80";

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, str, 1, &s) == -2);
  assert (mbrtowc (&wc, str + 1, 2, &s) == 2);
  assert (wc == 0x800);

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, str, 3, &s) == 3);
  assert (wc == 0x800);

  return 0;
}

/* Test for NUL byte processing via empty string.  */
static int
utf8_test_2 (void)
{
  wchar_t wc;
  mbstate_t s;

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (NULL, "", 1, &s) == 0); /* valid terminator */
  assert (mbsinit (&s));

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, "\xE2", 1, &s) == (size_t) -2);	/* 1st byte processed */
  assert (mbrtowc (NULL, "", 1, &s) == (size_t) -1); /* invalid terminator */

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, "\xE2", 1, &s) == (size_t) -2);	/* 1st byte processed */
  assert (mbrtowc (&wc, "\x89", 1, &s) == (size_t) -2);	/* 2nd byte processed */
  assert (mbrtowc (NULL, "", 1, &s) == (size_t) -1); /* invalid terminator */

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, "\xE2", 1, &s) == (size_t) -2);	/* 1st byte processed */
  assert (mbrtowc (&wc, "\x89", 1, &s) == (size_t) -2);	/* 2nd byte processed */
  assert (mbrtowc (&wc, "\xA0", 1, &s) == 1);	/* 3nd byte processed */
  assert (mbrtowc (NULL, "", 1, &s) == 0); /* valid terminator */
  assert (mbsinit (&s));

  return 0;
}

/* Test for NUL byte processing via NULL string.  */
static int
utf8_test_3 (void)
{
  wchar_t wc;
  mbstate_t s;

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (NULL, NULL, 0, &s) == 0); /* valid terminator */
  assert (mbsinit (&s));

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, "\xE2", 1, &s) == (size_t) -2);	/* 1st byte processed */
  assert (mbrtowc (NULL, NULL, 0, &s) == (size_t) -1); /* invalid terminator */

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, "\xE2", 1, &s) == (size_t) -2);	/* 1st byte processed */
  assert (mbrtowc (&wc, "\x89", 1, &s) == (size_t) -2);	/* 2nd byte processed */
  assert (mbrtowc (NULL, NULL, 0, &s) == (size_t) -1); /* invalid terminator */

  wc = 42;			/* arbitrary number */
  memset (&s, 0, sizeof (s));	/* get s into initial state */
  assert (mbrtowc (&wc, "\xE2", 1, &s) == (size_t) -2);	/* 1st byte processed */
  assert (mbrtowc (&wc, "\x89", 1, &s) == (size_t) -2);	/* 2nd byte processed */
  assert (mbrtowc (&wc, "\xA0", 1, &s) == 1);	/* 3nd byte processed */
  assert (mbrtowc (NULL, NULL, 0, &s) == 0); /* valid terminator */
  assert (mbsinit (&s));

  return 0;
}

static int
utf8_test (void)
{
  const char *locale = "de_DE.UTF-8";
  int error = 0;

  if (!setlocale (LC_CTYPE, locale))
    {
      fprintf (stderr, "locale '%s' not available!\n", locale);
      exit (1);
    }

  error |= utf8_test_1 ();
  error |= utf8_test_2 ();
  error |= utf8_test_3 ();

  return error;
}


static int
do_test (void)
{
  int result = 0;

  /* Check mapping of ASCII range for some character sets which have
     ASCII as a subset.  For those the wide char generated must have
     the same value.  */
  setlocale (LC_ALL, "C");
  result |= check_ascii (setlocale (LC_ALL, NULL));

  setlocale (LC_ALL, "de_DE.UTF-8");
  result |= check_ascii (setlocale (LC_ALL, NULL));
  result |= utf8_test ();

  setlocale (LC_ALL, "ja_JP.EUC-JP");
  result |= check_ascii (setlocale (LC_ALL, NULL));

  return result;
}


static int
check_ascii (const char *locname)
{
  int c;
  int res = 0;

  printf ("Testing locale \"%s\":\n", locname);

  for (c = 0; c <= 127; ++c)
    {
      char buf[MB_CUR_MAX];
      wchar_t wc = 0xffffffff;
      mbstate_t s;
      size_t n, i;

      for (i = 0; i < MB_CUR_MAX; ++i)
	buf[i] = c + i;

      memset (&s, '\0', sizeof (s));

      n = mbrtowc (&wc, buf, MB_CUR_MAX, &s);
      if (n == (size_t) -1)
	{
	  printf ("%s: '\\x%x': encoding error\n", locname, c);
	  ++res;
	}
      else if (n == (size_t) -2)
	{
	  printf ("%s: '\\x%x': incomplete character\n", locname, c);
	  ++res;
	}
      else if (n == 0 && c != 0)
	{
	  printf ("%s: '\\x%x': 0 returned\n", locname, c);
	  ++res;
	}
      else if (n != 0 && c == 0)
	{
	  printf ("%s: '\\x%x': not 0 returned\n", locname, c);
	  ++res;
	}
      else if (c != 0 && n != 1)
	{
	  printf ("%s: '\\x%x': not 1 returned\n", locname, c);
	  ++res;
	}
      else if (wc != (wchar_t) c)
	{
	  printf ("%s: '\\x%x': wc != L'\\x%x'\n", locname, c, c);
	  ++res;
	}
    }

  printf (res == 1 ? "%d error\n" : "%d errors\n", res);

  return res != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
