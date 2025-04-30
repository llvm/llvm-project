/* Measure strlen functions.
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
# define TEST_NAME "strnlen"
#else
# define TEST_NAME "wcsnlen"
# define generic_strnlen generic_wcsnlen
# define memchr_strnlen wcschr_wcsnlen
#endif /* WIDE */
#include "bench-string.h"

#define BIG_CHAR MAX_CHAR

#ifndef WIDE
# define MIDDLE_CHAR 127
#else
# define MIDDLE_CHAR 1121
#endif /* WIDE */

typedef size_t (*proto_t) (const CHAR *, size_t);
size_t generic_strnlen (const CHAR *, size_t);

size_t
memchr_strnlen (const CHAR *s, size_t maxlen)
{
  const CHAR *s1 = MEMCHR (s, 0, maxlen);
  return (s1 == NULL) ? maxlen : s1 - s;
}

IMPL (STRNLEN, 1)
IMPL (memchr_strnlen, 0)
IMPL (generic_strnlen, 0)

static void
do_one_test (impl_t *impl, const CHAR *s, size_t maxlen, size_t exp_len)
{
  size_t len = CALL (impl, s, maxlen), i, iters = INNER_LOOP_ITERS_LARGE;
  timing_t start, stop, cur;

  if (len != exp_len)
    {
      error (0, 0, "Wrong result in function %s %zd %zd", impl->name,
	     len, exp_len);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s, maxlen);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align, size_t len, size_t maxlen, int max_char)
{
  size_t i;

  align &= 63;
  if ((align + len) * sizeof (CHAR) >= page_size)
    return;

  CHAR *buf = (CHAR *) (buf1);

  for (i = 0; i < len; ++i)
    buf[align + i] = 1 + 7 * i % max_char;
  buf[align + len] = 0;

  printf ("Length %4zd, alignment %2zd:", len, align);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, (CHAR *) (buf + align), maxlen, MIN (len, maxlen));

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
      do_test (0, i, i - 1, MIDDLE_CHAR);
      do_test (0, i, i, MIDDLE_CHAR);
      do_test (0, i, i + 1, MIDDLE_CHAR);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, i, i - 1, MIDDLE_CHAR);
      do_test (i, i, i, MIDDLE_CHAR);
      do_test (i, i, i + 1, MIDDLE_CHAR);
    }

  for (i = 2; i <= 10; ++i)
    {
      do_test (0, 1 << i, 5000, MIDDLE_CHAR);
      do_test (1, 1 << i, 5000, MIDDLE_CHAR);
    }

  for (i = 1; i < 8; ++i)
    do_test (0, i, 5000, BIG_CHAR);

  for (i = 1; i < 8; ++i)
    do_test (i, i, 5000, BIG_CHAR);

  for (i = 2; i <= 10; ++i)
    {
      do_test (0, 1 << i, 5000, BIG_CHAR);
      do_test (1, 1 << i, 5000, BIG_CHAR);
    }

  return ret;
}

#include <support/test-driver.c>

#define libc_hidden_def(X)
#ifndef WIDE
# undef STRNLEN
# define STRNLEN generic_strnlen
# include <string/strnlen.c>
#else
# define WCSNLEN generic_strnlen
# include <wcsmbs/wcsnlen.c>
#endif
