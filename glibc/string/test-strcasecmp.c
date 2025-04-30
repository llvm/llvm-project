/* Test and measure strcasecmp functions.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Jakub Jelinek <jakub@redhat.com>, 1999.

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

#include <locale.h>
#include <ctype.h>
#define TEST_MAIN
#define TEST_NAME "strcasecmp"
#include "test-string.h"

typedef int (*proto_t) (const char *, const char *);
static int simple_strcasecmp (const char *, const char *);
static int stupid_strcasecmp (const char *, const char *);

IMPL (stupid_strcasecmp, 0)
IMPL (simple_strcasecmp, 0)
IMPL (strcasecmp, 1)

static int
simple_strcasecmp (const char *s1, const char *s2)
{
  int ret;

  while ((ret = ((unsigned char) tolower (*s1)
		 - (unsigned char) tolower (*s2))) == 0
	 && *s1++)
    ++s2;
  return ret;
}

static int
stupid_strcasecmp (const char *s1, const char *s2)
{
  size_t ns1 = strlen (s1) + 1, ns2 = strlen (s2) + 1;
  size_t n = ns1 < ns2 ? ns1 : ns2;
  int ret = 0;

  while (n--)
    {
      if ((ret = ((unsigned char) tolower (*s1)
		  - (unsigned char) tolower (*s2))) != 0)
	break;
      ++s1;
      ++s2;
    }
  return ret;
}

static void
do_one_test (impl_t *impl, const char *s1, const char *s2, int exp_result)
{
  int result = CALL (impl, s1, s2);
  if ((exp_result == 0 && result != 0)
      || (exp_result < 0 && result >= 0)
      || (exp_result > 0 && result <= 0))
    {
      error (0, 0, "Wrong result in function %s %d %d", impl->name,
	     result, exp_result);
      ret = 1;
      return;
    }
}

static void
do_test (size_t align1, size_t align2, size_t len, int max_char,
	 int exp_result)
{
  size_t i;
  char *s1, *s2;

  if (len == 0)
    return;

  align1 &= 7;
  if (align1 + len + 1 >= page_size)
    return;

  align2 &= 7;
  if (align2 + len + 1 >= page_size)
    return;

  s1 = (char *) (buf1 + align1);
  s2 = (char *) (buf2 + align2);

  for (i = 0; i < len; i++)
    {
      s1[i] = toupper (1 + 23 * i % max_char);
      s2[i] = tolower (s1[i]);
    }

  s1[len] = s2[len] = 0;
  s1[len + 1] = 23;
  s2[len + 1] = 24 + exp_result;
  if ((s2[len - 1] == 'z' && exp_result == -1)
      || (s2[len - 1] == 'a' && exp_result == 1))
    s1[len - 1] += exp_result;
  else
    s2[len - 1] -= exp_result;

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s1, s2, exp_result);
}

static void
do_random_tests (void)
{
  size_t i, j, n, align1, align2, pos, len1, len2;
  int result;
  long r;
  unsigned char *p1 = buf1 + page_size - 512;
  unsigned char *p2 = buf2 + page_size - 512;

  for (n = 0; n < ITERATIONS; n++)
    {
      align1 = random () & 31;
      if (random () & 1)
	align2 = random () & 31;
      else
	align2 = align1 + (random () & 24);
      pos = random () & 511;
      j = align1 > align2 ? align1 : align2;
      if (pos + j >= 511)
	pos = 510 - j - (random () & 7);
      len1 = random () & 511;
      if (pos >= len1 && (random () & 1))
	len1 = pos + (random () & 7);
      if (len1 + j >= 512)
	len1 = 511 - j - (random () & 7);
      if (pos >= len1)
	len2 = len1;
      else
	len2 = len1 + (len1 != 511 - j ? random () % (511 - j - len1) : 0);
      j = (pos > len2 ? pos : len2) + align1 + 64;
      if (j > 512)
	j = 512;
      for (i = 0; i < j; ++i)
	{
	  p1[i] = tolower (random () & 255);
	  if (i < len1 + align1 && !p1[i])
	    {
	      p1[i] = tolower (random () & 255);
	      if (!p1[i])
		p1[i] = tolower (1 + (random () & 127));
	    }
	}
      for (i = 0; i < j; ++i)
	{
	  p2[i] = toupper (random () & 255);
	  if (i < len2 + align2 && !p2[i])
	    {
	      p2[i] = toupper (random () & 255);
	      if (!p2[i])
		toupper (p2[i] = 1 + (random () & 127));
	    }
	}

      result = 0;
      memcpy (p2 + align2, p1 + align1, pos);
      if (pos < len1)
	{
	  if (tolower (p2[align2 + pos]) == p1[align1 + pos])
	    {
	      p2[align2 + pos] = toupper (random () & 255);
	      if (tolower (p2[align2 + pos]) == p1[align1 + pos])
		p2[align2 + pos] = toupper (p1[align1 + pos]
					    + 3 + (random () & 127));
	    }

	  if (p1[align1 + pos] < tolower (p2[align2 + pos]))
	    result = -1;
	  else
	    result = 1;
	}
      p1[len1 + align1] = 0;
      p2[len2 + align2] = 0;

      FOR_EACH_IMPL (impl, 1)
	{
	  r = CALL (impl, (char *) (p1 + align1), (char *) (p2 + align2));
	  /* Test whether on 64-bit architectures where ABI requires
	     callee to promote has the promotion been done.  */
	  asm ("" : "=g" (r) : "0" (r));
	  if ((r == 0 && result)
	      || (r < 0 && result >= 0)
	      || (r > 0 && result <= 0))
	    {
	      error (0, 0, "Iteration %zd - wrong result in function %s (%zd, %zd, %zd, %zd, %zd) %ld != %d, p1 %p p2 %p",
		     n, impl->name, align1, align2, len1, len2, pos, r, result, p1, p2);
	      ret = 1;
	    }
	}
    }
}

static void
test_locale (const char *locale)
{
  size_t i;

  if (setlocale (LC_CTYPE, locale) == NULL)
    {
      error (0, 0, "cannot set locale \"%s\"", locale);
      ret = 1;
    }

  printf ("%-23s", locale);
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 1; i < 16; ++i)
    {
      do_test (i, i, i, 127, 0);
      do_test (i, i, i, 127, 1);
      do_test (i, i, i, 127, -1);
    }

  for (i = 1; i < 10; ++i)
    {
      do_test (0, 0, 2 << i, 127, 0);
      do_test (0, 0, 2 << i, 254, 0);
      do_test (0, 0, 2 << i, 127, 1);
      do_test (0, 0, 2 << i, 254, 1);
      do_test (0, 0, 2 << i, 127, -1);
      do_test (0, 0, 2 << i, 254, -1);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 2 * i, 8 << i, 127, 0);
      do_test (2 * i, i, 8 << i, 254, 0);
      do_test (i, 2 * i, 8 << i, 127, 1);
      do_test (2 * i, i, 8 << i, 254, 1);
      do_test (i, 2 * i, 8 << i, 127, -1);
      do_test (2 * i, i, 8 << i, 254, -1);
    }

  do_random_tests ();
}

int
test_main (void)
{
  test_init ();

  test_locale ("C");
  test_locale ("en_US.ISO-8859-1");
  test_locale ("en_US.UTF-8");
  test_locale ("tr_TR.ISO-8859-9");
  test_locale ("tr_TR.UTF-8");

  return ret;
}

#include <support/test-driver.c>
