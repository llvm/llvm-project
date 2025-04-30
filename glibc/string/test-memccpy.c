/* Test and measure memccpy functions.
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

#define TEST_MAIN
#define TEST_NAME "memccpy"
#include "test-string.h"

void *simple_memccpy (void *, const void *, int, size_t);
void *stupid_memccpy (void *, const void *, int, size_t);

IMPL (stupid_memccpy, 0)
IMPL (simple_memccpy, 0)
IMPL (memccpy, 1)

void *
simple_memccpy (void *dst, const void *src, int c, size_t n)
{
  const char *s = src;
  char *d = dst;

  while (n-- > 0)
    if ((*d++ = *s++) == (char) c)
      return d;

  return NULL;
}

void *
stupid_memccpy (void *dst, const void *src, int c, size_t n)
{
  void *p = memchr (src, c, n);

  if (p != NULL)
    return mempcpy (dst, src, p - src + 1);

  memcpy (dst, src, n);
  return NULL;
}

typedef void *(*proto_t) (void *, const void *, int c, size_t);

static void
do_one_test (impl_t *impl, void *dst, const void *src, int c, size_t len,
	     size_t n)
{
  void *expect = len > n ? NULL : (char *) dst + len;
  if (CALL (impl, dst, src, c, n) != expect)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     CALL (impl, dst, src, c, n), expect);
      ret = 1;
      return;
    }

  if (memcmp (dst, src, len > n ? n : len) != 0)
    {
      error (0, 0, "Wrong result in function %s", impl->name);
      ret = 1;
      return;
    }
}

static void
do_test (size_t align1, size_t align2, int c, size_t len, size_t n,
	 int max_char)
{
  size_t i;
  char *s1, *s2;

  align1 &= 7;
  if (align1 + len >= page_size)
    return;

  align2 &= 7;
  if (align2 + len >= page_size)
    return;

  s1 = (char *) (buf1 + align1);
  s2 = (char *) (buf2 + align2);

  for (i = 0; i < len - 1; ++i)
    {
      s1[i] = 32 + 23 * i % (max_char - 32);
      if (s1[i] == (char) c)
	--s1[i];
    }
  s1[len - 1] = c;
  for (i = len; i + align1 < page_size && i < len + 64; ++i)
    s1[i] = 32 + 32 * i % (max_char - 32);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s2, s1, c, len, n);
}

static void
do_random_tests (void)
{
  size_t i, j, n, align1, align2, len, size, mode;
  unsigned char *p1 = buf1 + page_size - 512;
  unsigned char *p2 = buf2 + page_size - 512;
  unsigned char *res, c;

  for (n = 0; n < ITERATIONS; n++)
    {
      mode = random ();
      c = random ();
      if (mode & 1)
	{
	  size = random () & 255;
	  align1 = 512 - size - (random () & 15);
	  if (mode & 2)
	    align2 = align1 - (random () & 24);
	  else
	    align2 = align1 - (random () & 31);
	  if (mode & 4)
	    {
	      j = align1;
	      align1 = align2;
	      align2 = j;
	    }
	  if (mode & 8)
	    len = size - (random () & 31);
	  else
	    len = 512;
	  if (len >= 512)
	    len = random () & 511;
	}
      else
	{
	  align1 = random () & 31;
	  if (mode & 2)
	    align2 = random () & 31;
	  else
	    align2 = align1 + (random () & 24);
	  len = random () & 511;
	  j = align1;
	  if (align2 > j)
	    j = align2;
	  if (mode & 4)
	    {
	      size = random () & 511;
	      if (size + j > 512)
		size = 512 - j - (random() & 31);
	    }
	  else
	    size = 512 - j;
	  if ((mode & 8) && len + j >= 512)
	    len = 512 - j - (random () & 7);
	}
      j = len + align1 + 64;
      if (j > 512)
	j = 512;
      for (i = 0; i < j; i++)
	{
	  if (i == len + align1)
	    p1[i] = c;
	  else
	    {
	      p1[i] = random () & 255;
	      if (i >= align1 && i < len + align1 && p1[i] == c)
		p1[i] = (random () & 127) + 3 + c;
	    }
	}

      FOR_EACH_IMPL (impl, 1)
	{
	  unsigned char *expect;
	  memset (p2 - 64, '\1', 512 + 64);
	  res = CALL (impl, p2 + align2, p1 + align1, (char) c, size);
	  if (len >= size)
	    expect = NULL;
	  else
	    expect = p2 + align2 + len + 1;

	  if (res != expect)
	    {
	      error (0, 0, "Iteration %zd - wrong result in function %s (%zd, %zd, %zd, %zd, %d) %p != %p",
		     n, impl->name, align1, align2, len, size, c, res, expect);
	      ret = 1;
	    }
	  for (j = 0; j < align2 + 64; ++j)
	    {
	      if (p2[j - 64] != '\1')
		{
		  error (0, 0, "Iteration %zd - garbage before, %s (%zd, %zd, %zd)",
			 n, impl->name, align1, align2, len);
		  ret = 1;
		  break;
		}
	    }
	  j = align2 + len + 1;
	  if (size + align2 < j)
	    j = size + align2;
	  for (; j < 512; ++j)
	    {
	      if (p2[j] != '\1')
		{
		  error (0, 0, "Iteration %zd - garbage after, %s (%zd, %zd, %zd)",
			 n, impl->name, align1, align2, len);
		  ret = 1;
		  break;
		}
	    }
	  j = len + 1;
	  if (size < j)
	    j = size;
	  if (memcmp (p1 + align1, p2 + align2, j))
	    {
	      error (0, 0, "Iteration %zd - different strings, %s (%zd, %zd, %zd)",
		     n, impl->name, align1, align2, len);
	      ret = 1;
	    }
	}
    }
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

  for (i = 1; i < 8; ++i)
    {
      do_test (i, i, 12, 16, 16, 127);
      do_test (i, i, 23, 16, 16, 255);
      do_test (i, 2 * i, 28, 16, 16, 127);
      do_test (2 * i, i, 31, 16, 16, 255);
      do_test (8 - i, 2 * i, 1, 1 << i, 2 << i, 127);
      do_test (2 * i, 8 - i, 17, 2 << i, 1 << i, 127);
      do_test (8 - i, 2 * i, 0, 1 << i, 2 << i, 255);
      do_test (2 * i, 8 - i, i, 2 << i, 1 << i, 255);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, i, 4 << i, 8 << i, 127);
      do_test (0, 0, i, 16 << i, 8 << i, 127);
      do_test (8 - i, 2 * i, i, 4 << i, 8 << i, 127);
      do_test (8 - i, 2 * i, i, 16 << i, 8 << i, 127);
    }

  do_random_tests ();
  return ret;
}

#include <support/test-driver.c>
