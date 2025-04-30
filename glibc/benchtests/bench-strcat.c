/* Measure strcat functions.
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
#ifndef WIDE
# define TEST_NAME "strcat"
#else
# define TEST_NAME "wcscat"
# define generic_strcat generic_wcscat
#endif /* WIDE */
#include "bench-string.h"

#define BIG_CHAR MAX_CHAR

#ifndef WIDE
# define sfmt "s"
# define SMALL_CHAR 127
#else
# define sfmt "ls"
# define SMALL_CHAR 1273
#endif /* WIDE */


typedef CHAR *(*proto_t) (CHAR *, const CHAR *);

CHAR *
generic_strcat (CHAR *dst, const CHAR *src)
{
  STRCPY (dst + STRLEN (dst), src);
  return dst;
}

IMPL (STRCAT, 1)
IMPL (generic_strcat, 0)

static void
do_one_test (impl_t *impl, CHAR *dst, const CHAR *src)
{
  size_t k = STRLEN (dst), i, iters = INNER_LOOP_ITERS_LARGE;
  timing_t start, stop, cur;

  if (CALL (impl, dst, src) != dst)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     CALL (impl, dst, src), dst);
      ret = 1;
      return;
    }

  if (STRCMP (dst + k, src) != 0)
    {
      error (0, 0, "Wrong result in function %s dst \"%" sfmt "\" src \"%" sfmt "\"",
	     impl->name, dst, src);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      dst[k] = '\0';
      CALL (impl, dst, src);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align1, size_t align2, size_t len1, size_t len2, int max_char)
{
  size_t i;
  CHAR *s1, *s2;

  align1 &= 7;
  if ((align1 + len1) * sizeof (CHAR) >= page_size)
    return;

  align2 &= 7;
  if ((align2 + len1 + len2) * sizeof (CHAR) >= page_size)
    return;

  s1 = (CHAR *) (buf1) + align1;
  s2 = (CHAR *) (buf2) + align2;

  for (i = 0; i < len1; ++i)
    s1[i] = 32 + 23 * i % (max_char - 32);
  s1[len1] = '\0';

  for (i = 0; i < len2; i++)
    s2[i] = 32 + 23 * i % (max_char - 32);

  printf ("Length %4zd/%4zd, alignment %2zd/%2zd:", len1, len2, align1, align2);

  FOR_EACH_IMPL (impl, 0)
    {
      s2[len2] = '\0';
      do_one_test (impl, s2, s1);
    }

  putchar ('\n');
}

int
test_main (void)
{
  size_t i;

  test_init ();

  printf ("%28s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 0; i < 16; ++i)
    {
      do_test (0, 0, i, i, SMALL_CHAR);
      do_test (0, 0, i, i, BIG_CHAR);
      do_test (0, i, i, i, SMALL_CHAR);
      do_test (i, 0, i, i, BIG_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, 8 << i, 8 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 8 << i, 8 << i, SMALL_CHAR);
      do_test (0, 0, 8 << i, 2 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 8 << i, 2 << i, SMALL_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 2 * i, 8 << i, 1, SMALL_CHAR);
      do_test (2 * i, i, 8 << i, 1, BIG_CHAR);
      do_test (i, i, 8 << i, 10, SMALL_CHAR);
      do_test (i, i, 8 << i, 10, BIG_CHAR);
    }

  return ret;
}

#include <support/test-driver.c>
