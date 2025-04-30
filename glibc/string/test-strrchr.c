/* Test and measure STRCHR functions.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Jakub Jelinek <jakub@redhat.com>, 1999.
   Added wcsrrchr support by Liubov Dmitrieva <liubov.dmitrieva@gmail.com>,
   2011.

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
#ifdef WIDE
# define TEST_NAME "wcsrchr"
#else
# define TEST_NAME "strrchr"
#endif
#include "test-string.h"

#ifdef WIDE
# include <wchar.h>
# define SIMPLE_STRRCHR simple_wcsrchr
# define STRRCHR wcsrchr
# define CHAR wchar_t
# define UCHAR wchar_t
# define BIG_CHAR WCHAR_MAX
# define SMALL_CHAR 1273
#else
# define SIMPLE_STRRCHR simple_strrchr
# define STRRCHR strrchr
# define CHAR char
# define UCHAR unsigned char
# define BIG_CHAR CHAR_MAX
# define SMALL_CHAR 127
#endif

typedef CHAR *(*proto_t) (const CHAR *, int);
CHAR *SIMPLE_STRRCHR (const CHAR *, int);

IMPL (SIMPLE_STRRCHR, 0)
IMPL (STRRCHR, 1)

CHAR *
SIMPLE_STRRCHR (const CHAR *s, int c)
{
  const CHAR *ret = NULL;

  for (; *s != '\0'; ++s)
    if (*s == (CHAR) c)
      ret = s;

  return (CHAR *) (c == '\0' ? s : ret);
}

static void
do_one_test (impl_t *impl, const CHAR *s, int c, CHAR *exp_res)
{
  CHAR *res = CALL (impl, s, c);
  if (res != exp_res)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     res, exp_res);
      ret = 1;
      return;
    }
}

static void
do_test (size_t align, size_t pos, size_t len, int seek_char, int max_char)
/* For wcsrchr: align here means align not in bytes,
   but in wchar_ts, in bytes it will equal to align * (sizeof (wchar_t))
   len for wcschr here isn't in bytes but it's number of wchar_t symbols.  */
{
  size_t i;
  CHAR *result;
  CHAR *buf = (CHAR *) buf1;

  align &= 7;
  if ( (align + len) * sizeof (CHAR) >= page_size)
    return;

  for (i = 0; i < len; ++i)
    {
      buf[align + i] = (random () * random ()) & max_char;
      if (!buf[align + i])
	buf[align + i] = (random () * random ()) & max_char;
      if (!buf[align + i])
	buf[align + i] = 1;
      if ((i > pos || pos >= len) && buf[align + i] == seek_char)
	buf[align + i] = seek_char + 10 + (random () & 15);
    }
  buf[align + len] = 0;

  if (pos < len)
    {
      buf[align + pos] = seek_char;
      result = (CHAR *) (buf + align + pos);
    }
  else if (seek_char == 0)
    result = (CHAR *) (buf + align + len);
  else
    result = NULL;

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, (CHAR *) (buf + align), seek_char, result);
}

static void
do_random_tests (void)
{
  size_t i, j, n, align, pos, len;
  int seek_char;
  CHAR *result;
  UCHAR *p = (UCHAR *) (buf1 + page_size) - 512;

  for (n = 0; n < ITERATIONS; n++)
    {
      align = random () & (63 / sizeof (CHAR));
      /* For wcsrchr: align here means align not in bytes, but in wchar_ts,
	 in bytes it will equal to align * (sizeof (wchar_t)).
	 For strrchr we need to check all alignments from 0 to 63 since
	 some assembly implementations have separate prolog for alignments
	 more 48. */
      pos = random () & 511;
      if (pos + align >= 511)
	pos = 510 - align - (random () & 7);
      len = random () & 511;
      /* len for wcschr here isn't in bytes but it's number of wchar_t
	 symbols.  */
      if (pos >= len)
	len = pos + (random () & 7);
      if (len + align >= 512)
	len = 511 - align - (random () & 7);
      seek_char = random () & 255;
      if (seek_char && pos == len)
	{
	  if (pos)
	    --pos;
	  else
	    ++len;
	}
      j = len + align + 64;
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
	      if (((i > pos + align && i < len + align) || pos > len)
		  && p[i] == seek_char)
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
	result = NULL;

      FOR_EACH_IMPL (impl, 1)
	if (CALL (impl, (CHAR *) (p + align), seek_char) != result)
	  {
	    error (0, 0, "Iteration %zd - wrong result in function %s (%zd, %d, %zd, %zd) %p != %p, p %p",
		   n, impl->name, align, seek_char, len, pos,
		   CALL (impl, (CHAR *) (p + align), seek_char), result, p);
	    ret = 1;
	  }
    }
}

int
test_main (void)
{
  size_t i;

  test_init ();

  printf ("%20s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 16 << i, 2048, 23, SMALL_CHAR);
      do_test (i, 16 << i, 2048, 23, SMALL_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 64, 256, 23, SMALL_CHAR);
      do_test (i, 64, 256, 23, BIG_CHAR);
    }

  for (i = 0; i < 32; ++i)
    {
      do_test (0, i, i + 1, 23, SMALL_CHAR);
      do_test (0, i, i + 1, 23, BIG_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 16 << i, 2048, 0, SMALL_CHAR);
      do_test (i, 16 << i, 2048, 0, SMALL_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 64, 256, 0, SMALL_CHAR);
      do_test (i, 64, 256, 0, BIG_CHAR);
    }

  for (i = 0; i < 32; ++i)
    {
      do_test (0, i, i + 1, 0, SMALL_CHAR);
      do_test (0, i, i + 1, 0, BIG_CHAR);
    }

  do_random_tests ();
  return ret;
}

#include <support/test-driver.c>
