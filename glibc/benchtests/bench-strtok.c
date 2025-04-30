/* Measure strtok functions.
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
#define TEST_NAME "strtok"
#include "bench-string.h"

char *
oldstrtok (char *s, const char *delim)
{
  static char *olds;
  char *token;

  if (s == NULL)
    s = olds;

  /* Scan leading delimiters.  */
  s += strspn (s, delim);
  if (*s == '\0')
    {
      olds = s;
      return NULL;
    }

  /* Find the end of the token.  */
  token = s;
  s = strpbrk (token, delim);
  if (s == NULL)
    /* This token finishes the string.  */
    olds = rawmemchr (token, '\0');
  else
    {
      /* Terminate the token and make OLDS point past it.  */
      *s = '\0';
      olds = s + 1;
    }
  return token;
}

typedef char *(*proto_t) (const char *, const char *);

IMPL (oldstrtok, 0)
IMPL (strtok, 1)

static void
do_one_test (impl_t * impl, const char *s1, const char *s2)
{
  size_t i, iters = INNER_LOOP_ITERS_SMALL;
  timing_t start, stop, cur;
  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s1, s2);
      CALL (impl, NULL, s2);
      CALL (impl, NULL, s2);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);

}


static void
do_test (size_t align1, size_t align2, size_t len1, size_t len2, int fail)
{
  char *s2 = (char *) (buf2 + align2);
  static const char d[] = "1234567890abcdef";
#define dl (sizeof (d) - 1)
  char *ss2 = s2;
  for (size_t l = len2; l > 0; l = l > dl ? l - dl : 0)
    {
      size_t t = l > dl ? dl : l;
      ss2 = mempcpy (ss2, d, t);
    }
  s2[len2] = '\0';

  printf ("Length %4zd/%zd, alignment %2zd/%2zd, %s:",
	  len1, len2, align1, align2, fail ? "fail" : "found");

  FOR_EACH_IMPL (impl, 0)
  {
    char *s1 = (char *) (buf1 + align1);
    if (fail)
      {
	char *ss1 = s1;
	for (size_t l = len1; l > 0; l = l > dl ? l - dl : 0)
	  {
	    size_t t = l > dl ? dl : l;
	    memcpy (ss1, d, t);
	    ++ss1[len2 > 7 ? 7 : len2 - 1];
	    ss1 += t;
	  }
      }
    else
      {
	memset (s1, '0', len1);
	memcpy (s1 + (len1 - len2) - 2, s2, len2);
	if ((len1 / len2) > 4)
	  memcpy (s1 + (len1 - len2) - (3 * len2), s2, len2);
      }
    s1[len1] = '\0';
    do_one_test (impl, s1, s2);
  }
  putchar ('\n');
}

static int
test_main (void)
{
  test_init ();

  printf ("%23s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (size_t klen = 2; klen < 32; ++klen)
    for (size_t hlen = 2 * klen; hlen < 16 * klen; hlen += klen)
      {
	do_test (0, 0, hlen, klen, 0);
	do_test (0, 0, hlen, klen, 1);
	do_test (0, 3, hlen, klen, 0);
	do_test (0, 3, hlen, klen, 1);
	do_test (0, 9, hlen, klen, 0);
	do_test (0, 9, hlen, klen, 1);
	do_test (0, 15, hlen, klen, 0);
	do_test (0, 15, hlen, klen, 1);

	do_test (3, 0, hlen, klen, 0);
	do_test (3, 0, hlen, klen, 1);
	do_test (3, 3, hlen, klen, 0);
	do_test (3, 3, hlen, klen, 1);
	do_test (3, 9, hlen, klen, 0);
	do_test (3, 9, hlen, klen, 1);
	do_test (3, 15, hlen, klen, 0);
	do_test (3, 15, hlen, klen, 1);

	do_test (9, 0, hlen, klen, 0);
	do_test (9, 0, hlen, klen, 1);
	do_test (9, 3, hlen, klen, 0);
	do_test (9, 3, hlen, klen, 1);
	do_test (9, 9, hlen, klen, 0);
	do_test (9, 9, hlen, klen, 1);
	do_test (9, 15, hlen, klen, 0);
	do_test (9, 15, hlen, klen, 1);

	do_test (15, 0, hlen, klen, 0);
	do_test (15, 0, hlen, klen, 1);
	do_test (15, 3, hlen, klen, 0);
	do_test (15, 3, hlen, klen, 1);
	do_test (15, 9, hlen, klen, 0);
	do_test (15, 9, hlen, klen, 1);
	do_test (15, 15, hlen, klen, 0);
	do_test (15, 15, hlen, klen, 1);
      }
  do_test (0, 0, page_size - 1, 16, 0);
  do_test (0, 0, page_size - 1, 16, 1);

  return ret;
}

#include <support/test-driver.c>
