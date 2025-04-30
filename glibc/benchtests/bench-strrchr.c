/* Measure STRCHR functions.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
#include "bench-string.h"

#define BIG_CHAR MAX_CHAR

#ifdef WIDE
# define SIMPLE_STRRCHR simple_wcsrchr
# define SMALL_CHAR 1273
#else
# define SIMPLE_STRRCHR simple_strrchr
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
  size_t i, iters = INNER_LOOP_ITERS8;
  timing_t start, stop, cur;

  if (res != exp_res)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     res, exp_res);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s, c);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
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
  if ((align + len) * sizeof (CHAR) >= page_size)
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

  printf ("Length %4zd, alignment in bytes %2zd:", len, align * sizeof (CHAR));

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, (CHAR *) (buf + align), seek_char, result);

  putchar ('\n');
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

  return ret;
}

#include <support/test-driver.c>
