/* Measure strncpy functions.
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
# define SMALL_CHAR 1273
#else
# define SMALL_CHAR 127
#endif /* !WIDE */

#ifndef STRNCPY_RESULT
# define STRNCPY_RESULT(dst, len, n) dst
# define TEST_MAIN
# ifndef WIDE
#  define TEST_NAME "strncpy"
# else
#  define TEST_NAME "wcsncpy"
#  define generic_strncpy generic_wcsncpy
# endif /* WIDE */
# include "bench-string.h"

CHAR *
generic_strncpy (CHAR *dst, const CHAR *src, size_t n)
{
  size_t nc = STRNLEN (src, n);
  if (nc != n)
    MEMSET (dst + nc, 0, n - nc);
  return MEMCPY (dst, src, nc);
}

IMPL (STRNCPY, 1)
IMPL (generic_strncpy, 0)

#endif /* !STRNCPY_RESULT */

typedef CHAR *(*proto_t) (CHAR *, const CHAR *, size_t);

static void
do_one_test (impl_t *impl, CHAR *dst, const CHAR *src, size_t len, size_t n)
{
  size_t i, iters = INNER_LOOP_ITERS_LARGE * (4 / CHARBYTES);
  timing_t start, stop, cur;

  if (CALL (impl, dst, src, n) != STRNCPY_RESULT (dst, len, n))
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     CALL (impl, dst, src, n), dst);
      ret = 1;
      return;
    }

  if (memcmp (dst, src, (len > n ? n : len) * sizeof (CHAR)) != 0)
    {
      error (0, 0, "Wrong result in function %s", impl->name);
      ret = 1;
      return;
    }

  if (n > len)
    {
      size_t i;

      for (i = len; i < n; ++i)
	if (dst [i] != '\0')
	  {
	    error (0, 0, "Wrong result in function %s", impl->name);
	    ret = 1;
	    return;
	  }
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, dst, src, n);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align1, size_t align2, size_t len, size_t n, int max_char)
{
  size_t i;
  CHAR *s1, *s2;

/* For wcsncpy: align1 and align2 here mean alignment not in bytes,
   but in wchar_ts, in bytes it will equal to align * (sizeof (wchar_t)).  */
  align1 &= 7;
  if ((align1 + len) * sizeof (CHAR) >= page_size)
    return;

  align2 &= 7;
  if ((align2 + len) * sizeof (CHAR) >= page_size)
    return;

  s1 = (CHAR *) (buf1) + align1;
  s2 = (CHAR *) (buf2) + align2;

  for (i = 0; i < len; ++i)
    s1[i] = 32 + 23 * i % (max_char - 32);
  s1[len] = 0;
  for (i = len + 1; (i + align1) * sizeof (CHAR) < page_size && i < len + 64;
       ++i)
    s1[i] = 32 + 32 * i % (max_char - 32);

  printf ("Length %4zd, n %4zd, alignment %2zd/%2zd:", len, n, align1, align2);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s2, s1, len, n);

  putchar ('\n');
}

static int
test_main (void)
{
  size_t i;

  test_init ();

  printf ("%28s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 1; i < 8; ++i)
    {
      do_test (i, i, 16, 16, SMALL_CHAR);
      do_test (i, i, 16, 16, BIG_CHAR);
      do_test (i, 2 * i, 16, 16, SMALL_CHAR);
      do_test (2 * i, i, 16, 16, BIG_CHAR);
      do_test (8 - i, 2 * i, 1 << i, 2 << i, SMALL_CHAR);
      do_test (2 * i, 8 - i, 2 << i, 1 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 1 << i, 2 << i, BIG_CHAR);
      do_test (2 * i, 8 - i, 2 << i, 1 << i, BIG_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, 4 << i, 8 << i, SMALL_CHAR);
      do_test (0, 0, 16 << i, 8 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 4 << i, 8 << i, SMALL_CHAR);
      do_test (8 - i, 2 * i, 16 << i, 8 << i, SMALL_CHAR);
    }

  return ret;
}

#include <support/test-driver.c>
