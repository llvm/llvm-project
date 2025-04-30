/* Measure strpbrk functions.
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

#ifndef WIDE
# define SMALL_CHAR 127
#else
# define SMALL_CHAR 1273
#endif /* WIDE */

#ifndef STRPBRK_RESULT
# define STRPBRK_RESULT(s, pos) ((s)[(pos)] ? (s) + (pos) : NULL)
# define RES_TYPE CHAR *
# define TEST_MAIN
# ifndef WIDE
#  define TEST_NAME "strpbrk"
# else
#  define TEST_NAME "wcspbrk"
# endif /* WIDE */
# include "bench-string.h"

# ifndef WIDE
#  define SIMPLE_STRPBRK simple_strpbrk
# else
#  define SIMPLE_STRPBRK simple_wcspbrk
# endif /* WIDE */

typedef CHAR *(*proto_t) (const CHAR *, const CHAR *);
CHAR *SIMPLE_STRPBRK (const CHAR *, const CHAR *);

IMPL (SIMPLE_STRPBRK, 0)
IMPL (STRPBRK, 1)

CHAR *
SIMPLE_STRPBRK (const CHAR *s, const CHAR *rej)
{
  const CHAR *r;
  CHAR c;

  while ((c = *s++) != '\0')
    for (r = rej; *r != '\0'; ++r)
      if (*r == c)
	return (CHAR *) s - 1;
  return NULL;
}

#endif /* !STRPBRK_RESULT */

static void
do_one_test (impl_t *impl, const CHAR *s, const CHAR *rej, RES_TYPE exp_res)
{
  RES_TYPE res = CALL (impl, s, rej);
  size_t i, iters = INNER_LOOP_ITERS_MEDIUM;
  timing_t start, stop, cur;

  if (res != exp_res)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     (void *) res, (void *) exp_res);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s, rej);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align, size_t pos, size_t len)
{
  size_t i;
  int c;
  RES_TYPE result;
  CHAR *rej, *s;

  align &= 7;
  if ((align + pos + 10) * sizeof (CHAR) >= page_size || len > 240)
    return;

  rej = (CHAR *) (buf2) + (random () & 255);
  s = (CHAR *) (buf1) + align;

  for (i = 0; i < len; ++i)
    {
      rej[i] = random () & BIG_CHAR;
      if (!rej[i])
	rej[i] = random () & BIG_CHAR;
      if (!rej[i])
	rej[i] = 1 + (random () & SMALL_CHAR);
    }
  rej[len] = '\0';
  for (c = 1; c <= BIG_CHAR; ++c)
    if (STRCHR (rej, c) == NULL)
      break;

  for (i = 0; i < pos; ++i)
    {
      s[i] = random () & BIG_CHAR;
      if (STRCHR (rej, s[i]))
	{
	  s[i] = random () & BIG_CHAR;
	  if (STRCHR (rej, s[i]))
	    s[i] = c;
	}
    }
  s[pos] = rej[random () % (len + 1)];
  if (s[pos])
    {
      for (i = pos + 1; i < pos + 10; ++i)
	s[i] = random () & BIG_CHAR;
      s[i] = '\0';
    }
  result = STRPBRK_RESULT (s, pos);

  printf ("Length %4zd, alignment %2zd, rej len %2zd:", pos, align, len);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s, rej, result);

  putchar ('\n');
}

int
test_main (void)
{
  size_t i;

  test_init ();

  printf ("%32s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 0; i < 32; ++i)
    {
      do_test (0, 512, i);
      do_test (i, 512, i);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 16 << i, 4);
      do_test (i, 16 << i, 4);
    }

  for (i = 1; i < 8; ++i)
    do_test (i, 64, 10);

  for (i = 0; i < 64; ++i)
    do_test (0, i, 6);

  return ret;
}

#include <support/test-driver.c>
