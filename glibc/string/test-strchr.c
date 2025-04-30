/* Test STRCHR functions.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Jakub Jelinek <jakub@redhat.com>, 1999.
   Added wcschr support by Liubov Dmitrieva <liubov.dmitrieva@gmail.com>, 2011

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

#define TEST_MAIN
#ifndef WIDE
# ifdef USE_FOR_STRCHRNUL
#  define TEST_NAME "strchrnul"
# else
#  define TEST_NAME "strchr"
# endif /* !USE_FOR_STRCHRNUL */
#else
# ifdef USE_FOR_STRCHRNUL
#  define TEST_NAME "wcschrnul"
# else
#  define TEST_NAME "wcschr"
# endif /* !USE_FOR_STRCHRNUL */
#endif /* WIDE */
#include "test-string.h"

#ifndef WIDE
# ifdef USE_FOR_STRCHRNUL
#  define STRCHR strchrnul
#  define stupid_STRCHR stupid_STRCHRNUL
#  define simple_STRCHR simple_STRCHRNUL
# else
#  define STRCHR strchr
# endif /* !USE_FOR_STRCHRNUL */
# define STRLEN strlen
# define CHAR char
# define BIG_CHAR CHAR_MAX
# define MIDDLE_CHAR 127
# define SMALL_CHAR 23
# define UCHAR unsigned char
# define L(s) s
#else
# include <wchar.h>
# ifdef USE_FOR_STRCHRNUL
#  define STRCHR wcschrnul
#  define stupid_STRCHR stupid_WCSCHRNUL
#  define simple_STRCHR simple_WCSCHRNUL
# else
#  define STRCHR wcschr
# endif /* !USE_FOR_STRCHRNUL */
# define STRLEN wcslen
# define CHAR wchar_t
# define BIG_CHAR WCHAR_MAX
# define MIDDLE_CHAR 1121
# define SMALL_CHAR 851
# define UCHAR wchar_t
# define L(s) L ## s
#endif /* WIDE */

#ifdef USE_FOR_STRCHRNUL
# define NULLRET(endptr) endptr
#else
# define NULLRET(endptr) NULL
#endif /* !USE_FOR_STRCHRNUL */


typedef CHAR *(*proto_t) (const CHAR *, int);

CHAR *
simple_STRCHR (const CHAR *s, int c)
{
  for (; *s != (CHAR) c; ++s)
    if (*s == '\0')
      return NULLRET ((CHAR *) s);
  return (CHAR *) s;
}

CHAR *
stupid_STRCHR (const CHAR *s, int c)
{
  size_t n = STRLEN (s) + 1;

  while (n--)
    if (*s++ == (CHAR) c)
      return (CHAR *) s - 1;
  return NULLRET ((CHAR *) s - 1);
}

IMPL (stupid_STRCHR, 0)
IMPL (simple_STRCHR, 0)
IMPL (STRCHR, 1)

static int
check_result (impl_t *impl, const CHAR *s, int c, const CHAR *exp_res)
{
  CHAR *res = CALL (impl, s, c);
  if (res != exp_res)
    {
      error (0, 0, "Wrong result in function %s %#x %p %p", impl->name,
	     c, res, exp_res);
      ret = 1;
      return -1;
    }
  return 0;
}

static void
do_one_test (impl_t *impl, const CHAR *s, int c, const CHAR *exp_res)
{
  if (check_result (impl, s, c, exp_res) < 0)
    return;
}

static void
do_test (size_t align, size_t pos, size_t len, int seek_char, int max_char)
/* For wcschr: align here means align not in bytes,
   but in wchar_ts, in bytes it will equal to align * (sizeof (wchar_t))
   len for wcschr here isn't in bytes but it's number of wchar_t symbols.  */
{
  size_t i;
  CHAR *result;
  CHAR *buf = (CHAR *) buf1;
  align &= 127;
  if ((align + len) * sizeof (CHAR) >= page_size)
    return;

  for (i = 0; i < len; ++i)
    {
      buf[align + i] = 32 + 23 * i % max_char;
      if (buf[align + i] == seek_char)
	buf[align + i] = seek_char + 1;
      else if (buf[align + i] == 0)
	buf[align + i] = 1;
    }
  buf[align + len] = 0;

  if (pos < len)
    {
      buf[align + pos] = seek_char;
      result = buf + align + pos;
    }
  else if (seek_char == 0)
    result = buf + align + len;
  else
    result = NULLRET (buf + align + len);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, buf + align, seek_char, result);
}

static void
do_random_tests (void)
{
  size_t i, j, n, align, pos, len;
  int seek_char;
  CHAR *result;
  UCHAR *p = (UCHAR *) (buf1 + page_size - 512 * sizeof (CHAR));

  for (n = 0; n < ITERATIONS; n++)
    {
      /* For wcschr: align here means align not in bytes, but in wchar_ts,
	 in bytes it will equal to align * (sizeof (wchar_t)).  */
      align = random () & 15;
      pos = random () & 511;
      seek_char = random () & 255;
      if (pos + align >= 511)
	pos = 510 - align - (random () & 7);
      /* len for wcschr here isn't in bytes but it's number of wchar_t
	 symbols.  */
      len = random () & 511;
      if ((pos == len && seek_char)
	  || (pos > len && (random () & 1)))
	len = pos + 1 + (random () & 7);
      if (len + align >= 512)
	len = 511 - align - (random () & 7);
      if (pos == len && seek_char)
	len = pos + 1;
      j = (pos > len ? pos : len) + align + 64;
      if (j > 512)
	j = 512;

      for (i = 0; i < j; i++)
	{
	  if (i == pos + align)
	    p[i] = seek_char;
	  else if (i == len + align)
	    p[i] = 0;
	  else
	    {
	      p[i] = random () & 255;
	      if (i < pos + align && p[i] == seek_char)
		p[i] = seek_char + 13;
	      if (i < len + align && !p[i])
		{
		  p[i] = seek_char - 13;
		  if (!p[i])
		    p[i] = 140;
		}
	    }
	}

      if (pos <= len)
	result = (CHAR *) (p + pos + align);
      else if (seek_char == 0)
	result = (CHAR *) (p + len + align);
      else
	result = NULLRET ((CHAR *) (p + len + align));

      FOR_EACH_IMPL (impl, 1)
	if (CALL (impl, (CHAR *) (p + align), seek_char) != result)
	  {
	    error (0, 0, "Iteration %zd - wrong result in function \
		   %s (align in bytes: %zd, seek_char: %d, len: %zd, pos: %zd) %p != %p, p %p",
		   n, impl->name, align * sizeof (CHAR), seek_char, len, pos,
		   CALL (impl, (CHAR *) (p + align), seek_char), result, p);
	    ret = 1;
	  }
    }
}

static void
check1 (void)
{
  CHAR s[] __attribute__((aligned(16))) = L ("\xff");
  CHAR c = L ('\xfe');
  CHAR *exp_result = stupid_STRCHR (s, c);

  FOR_EACH_IMPL (impl, 0)
    check_result (impl, s, c, exp_result);
}

int
test_main (void)
{
  size_t i;

  test_init ();

  check1 ();

  printf ("%20s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 16 << i, 2048, SMALL_CHAR, MIDDLE_CHAR);
      do_test (i, 16 << i, 2048, SMALL_CHAR, MIDDLE_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 16 << i, 4096, SMALL_CHAR, MIDDLE_CHAR);
      do_test (i, 16 << i, 4096, SMALL_CHAR, MIDDLE_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 64, 256, SMALL_CHAR, MIDDLE_CHAR);
      do_test (i, 64, 256, SMALL_CHAR, BIG_CHAR);
    }

  for (i = 0; i < 8; ++i)
    {
      do_test (16 * i, 256, 512, SMALL_CHAR, MIDDLE_CHAR);
      do_test (16 * i, 256, 512, SMALL_CHAR, BIG_CHAR);
    }

  for (i = 0; i < 32; ++i)
    {
      do_test (0, i, i + 1, SMALL_CHAR, MIDDLE_CHAR);
      do_test (0, i, i + 1, SMALL_CHAR, BIG_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 16 << i, 2048, 0, MIDDLE_CHAR);
      do_test (i, 16 << i, 2048, 0, MIDDLE_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 16 << i, 4096, 0, MIDDLE_CHAR);
      do_test (i, 16 << i, 4096, 0, MIDDLE_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 64, 256, 0, MIDDLE_CHAR);
      do_test (i, 64, 256, 0, BIG_CHAR);
    }

  for (i = 0; i < 8; ++i)
    {
      do_test (16 * i, 256, 512, 0, MIDDLE_CHAR);
      do_test (16 * i, 256, 512, 0, BIG_CHAR);
    }

  for (i = 0; i < 32; ++i)
    {
      do_test (0, i, i + 1, 0, MIDDLE_CHAR);
      do_test (0, i, i + 1, 0, BIG_CHAR);
    }

  do_random_tests ();
  return ret;
}

#include <support/test-driver.c>
