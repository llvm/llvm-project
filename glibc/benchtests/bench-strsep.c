/* Measure strsep functions.
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
#define TEST_NAME "strsep"
#include "bench-string.h"

char *
simple_strsep (char **s1, char *s2)
{
  char *begin;
  char *s;
  size_t j = 0;

  begin = *s1;
  s = begin;
  if (begin == NULL)
    return NULL;
  ssize_t s2len = strlen (s2);
  while (*s)
    {
      for (j = 0; j < s2len; j++)
	{
	  if (*s == s2[j])
	    {
	      s[0] = '\0';
	      *s1 = s + 1;
	      return begin;
	    }
	}
      s++;
    }
  *s1 = NULL;
  return begin;
}

char *
oldstrsep (char **stringp, const char *delim)
{
  char *begin, *end;

  begin = *stringp;
  if (begin == NULL)
    return NULL;

  /* A frequent case is when the delimiter string contains only one
     character.  Here we don't need to call the expensive `strpbrk'
     function and instead work using `strchr'.  */
  if (delim[0] == '\0' || delim[1] == '\0')
    {
      char ch = delim[0];

      if (ch == '\0')
	end = NULL;
      else
	{
	  if (*begin == ch)
	    end = begin;
	  else if (*begin == '\0')
	    end = NULL;
	  else
	    end = strchr (begin + 1, ch);
	}
    }
  else
    /* Find the end of the token.  */
    end = strpbrk (begin, delim);

  if (end)
    {
      /* Terminate the token and set *STRINGP past NUL character.  */
      *end++ = '\0';
      *stringp = end;
    }
  else
    /* No more delimiters; this is the last token.  */
    *stringp = NULL;

  return begin;
}

typedef char *(*proto_t) (const char **, const char *);

IMPL (simple_strsep, 0)
IMPL (strsep, 1)
IMPL (oldstrsep, 2)

static void
do_one_test (impl_t * impl, const char *s1, const char *s2)
{
  size_t i, iters = INNER_LOOP_ITERS_SMALL;
  timing_t start, stop, cur;

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      const char *s1a = s1;
      CALL (impl, &s1a, s2);
      if (s1a != NULL)
	((char*)s1a)[-1] = '1';
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align1, size_t align2, size_t len1, size_t len2, int fail)
{
  char *s2 = (char *) (buf2 + align2);

  /* Search for a delimiter in a string containing mostly '0', so don't
     use '0' as a delimiter.  */
  static const char d[] = "123456789abcdefg";
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
    memset (s1, '0', len1);
    if (!fail)
      s1[len1 / 2] = '1';
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
    for (size_t hlen = 4 * klen; hlen < 8 * klen; hlen += klen)
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
