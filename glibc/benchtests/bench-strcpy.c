/* Measure strcpy functions.
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

#define BIG_CHAR MAX_CHAR

#ifdef WIDE
# define sfmt "ls"
# define SMALL_CHAR 1273
#else
# define sfmt "s"
# define SMALL_CHAR 127
#endif

#ifndef STRCPY_RESULT
# define STRCPY_RESULT(dst, len) dst
# define TEST_MAIN
# ifndef WIDE
#  define TEST_NAME "strcpy"
# else
#  define TEST_NAME "wcscpy"
#  define generic_strcpy generic_wcscpy
# endif
#include "bench-string.h"

CHAR *
generic_strcpy (CHAR *dst, const CHAR *src)
{
  return MEMCPY (dst, src, STRLEN (src) + 1);
}

IMPL (STRCPY, 1)
IMPL (generic_strcpy, 0)

#endif

typedef CHAR *(*proto_t) (CHAR *, const CHAR *);

static void
do_one_test (impl_t *impl, CHAR *dst, const CHAR *src,
	     size_t len __attribute__((unused)))
{
  size_t i, iters = INNER_LOOP_ITERS;
  timing_t start, stop, cur;

  if (CALL (impl, dst, src) != STRCPY_RESULT (dst, len))
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     CALL (impl, dst, src), STRCPY_RESULT (dst, len));
      ret = 1;
      return;
    }

  if (STRCMP (dst, src) != 0)
    {
      error (0, 0,
	     "Wrong result in function %s dst \"%" sfmt "\" src \"%" sfmt "\"",
	     impl->name, dst, src);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
	  CALL (impl, dst, src);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align1, size_t align2, size_t len, int max_char)
{
  size_t i;
  CHAR *s1, *s2;
/* For wcscpy: align1 and align2 here mean alignment not in bytes,
   but in wchar_ts, in bytes it will equal to align * (sizeof (wchar_t))
   len for wcschr here isn't in bytes but it's number of wchar_t symbols.  */
  align1 &= 7;
  if ((align1 + len) * sizeof (CHAR) >= page_size)
    return;

  align2 &= 7;
  if ((align2 + len) * sizeof (CHAR) >= page_size)
    return;

  s1 = (CHAR *) (buf1) + align1;
  s2 = (CHAR *) (buf2) + align2;

  for (i = 0; i < len; i++)
    s1[i] = 32 + 23 * i % (max_char - 32);
  s1[len] = 0;

  printf ("Length %4zd, alignments in bytes %2zd/%2zd:", len,
	  align1 * sizeof (CHAR), align2 * sizeof (CHAR));

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s2, s1, len);

  putchar ('\n');
}

int
test_main (void)
{
  size_t i;

  test_init ();

  printf ("%23s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 0; i < 16; ++i)
    {
      do_test (0, 0, i, SMALL_CHAR);
      do_test (0, 0, i, BIG_CHAR);
      do_test (0, i, i, SMALL_CHAR);
      do_test (i, 0, i, BIG_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, 8 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 8 << i, SMALL_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 2 * i, 8 << i, SMALL_CHAR);
      do_test (2 * i, i, 8 << i, BIG_CHAR);
      do_test (i, i, 8 << i, SMALL_CHAR);
      do_test (i, i, 8 << i, BIG_CHAR);
    }

  for (i = 16; i <= 512; i+=4)
    {
      do_test (0, 4, i, SMALL_CHAR);
      do_test (4, 0, i, BIG_CHAR);
      do_test (4, 4, i, SMALL_CHAR);
      do_test (2, 2, i, BIG_CHAR);
      do_test (2, 6, i, SMALL_CHAR);
      do_test (6, 2, i, BIG_CHAR);
      do_test (1, 7, i, SMALL_CHAR);
      do_test (7, 1, i, BIG_CHAR);
      do_test (3, 4, i, SMALL_CHAR);
      do_test (4, 3, i, BIG_CHAR);
      do_test (5, 7, i, SMALL_CHAR);
      do_test (7, 5, i, SMALL_CHAR);
    }

  return ret;
}

#include <support/test-driver.c>
