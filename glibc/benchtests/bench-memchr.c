/* Measure memchr functions.
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

#ifndef WIDE
# define SMALL_CHAR 127
#else
# define SMALL_CHAR 1273
#endif /* WIDE */

#ifndef USE_AS_MEMRCHR
# define TEST_MAIN
# ifndef WIDE
#  define TEST_NAME "memchr"
# else
#  define TEST_NAME "wmemchr"
# endif /* WIDE */
# include "bench-string.h"

# ifndef WIDE
#  define SIMPLE_MEMCHR simple_memchr
# else
#  define SIMPLE_MEMCHR simple_wmemchr
# endif /* WIDE */

typedef CHAR *(*proto_t) (const CHAR *, int, size_t);
CHAR *SIMPLE_MEMCHR (const CHAR *, int, size_t);

IMPL (SIMPLE_MEMCHR, 0)
IMPL (MEMCHR, 1)

CHAR *
SIMPLE_MEMCHR (const CHAR *s, int c, size_t n)
{
  while (n--)
    if (*s++ == (CHAR) c)
      return (CHAR *) s - 1;
  return NULL;
}
#endif /* !USE_AS_MEMRCHR */

static void
do_one_test (impl_t *impl, const CHAR *s, int c, size_t n)
{
  size_t i, iters = INNER_LOOP_ITERS_LARGE;
  timing_t start, stop, cur;

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s, c, n);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align, size_t pos, size_t len, int seek_char)
{
  size_t i;

  align &= 7;
  if ((align + len) * sizeof (CHAR) >= page_size)
    return;

  CHAR *buf = (CHAR *) (buf1);

  for (i = 0; i < len; ++i)
    {
      buf[align + i] = 1 + 23 * i % SMALL_CHAR;
      if (buf[align + i] == seek_char)
	buf[align + i] = seek_char + 1;
    }
  buf[align + len] = 0;

  if (pos < len)
    {
      buf[align + pos] = seek_char;
      buf[align + len] = -seek_char;
    }
  else
    {
      buf[align + len] = seek_char;
    }

  printf ("Length %4zd, position %4zd, alignment %2zd:",
	  len, pos, align);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, (CHAR *) (buf + align), seek_char, len);

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
      do_test (0, 16 << i, 2048, 23);
      do_test (i, 64, 256, 23);
      do_test (0, 16 << i, 2048, 0);
      do_test (i, 64, 256, 0);
#ifdef USE_AS_MEMRCHR
      /* Also test the position close to the beginning for memrchr.  */
      do_test (0, i, 256, 23);
      do_test (0, i, 256, 0);
      do_test (i, i, 256, 23);
      do_test (i, i, 256, 0);
#endif
    }
  for (i = 1; i < 8; ++i)
    {
      do_test (i, i << 5, 192, 23);
      do_test (i, i << 5, 192, 0);
      do_test (i, i << 5, 256, 23);
      do_test (i, i << 5, 256, 0);
      do_test (i, i << 5, 512, 23);
      do_test (i, i << 5, 512, 0);
    }
  for (i = 1; i < 32; ++i)
    {
      do_test (0, i, i + 1, 23);
      do_test (0, i, i + 1, 0);
      do_test (i, i, i + 1, 23);
      do_test (i, i, i + 1, 0);
      do_test (0, i, i - 1, 23);
      do_test (0, i, i - 1, 0);
      do_test (i, i, i - 1, 23);
      do_test (i, i, i - 1, 0);
#ifdef USE_AS_MEMRCHR
      /* Also test the position close to the beginning for memrchr.  */
      do_test (0, 1, i + 1, 23);
      do_test (0, 2, i + 1, 0);
#endif
    }

  return ret;
}

#include <support/test-driver.c>
