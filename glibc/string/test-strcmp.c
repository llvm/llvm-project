/* Test and measure strcmp and wcscmp functions.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Jakub Jelinek <jakub@redhat.com>, 1999.
   Added wcscmp support by Liubov Dmitrieva <liubov.dmitrieva@gmail.com>, 2011.

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
#ifdef WIDE
# define TEST_NAME "wcscmp"
#else
# define TEST_NAME "strcmp"
#endif
#include "test-string.h"

#ifdef WIDE
# include <wchar.h>

# define L(str) L##str
# define STRCMP wcscmp
# define STRCPY wcscpy
# define STRLEN wcslen
# define MEMCPY wmemcpy
# define SIMPLE_STRCMP simple_wcscmp
# define STUPID_STRCMP stupid_wcscmp
# define CHAR wchar_t
# define UCHAR wchar_t
# define CHARBYTES 4
# define CHARBYTESLOG 2
# define CHARALIGN __alignof__ (CHAR)
# define MIDCHAR 0x7fffffff
# define LARGECHAR 0xfffffffe
# define CHAR__MAX WCHAR_MAX
# define CHAR__MIN WCHAR_MIN

/* Wcscmp uses signed semantics for comparison, not unsigned */
/* Avoid using substraction since possible overflow */

int
simple_wcscmp (const wchar_t *s1, const wchar_t *s2)
{
  wchar_t c1, c2;
  do
    {
      c1 = *s1++;
      c2 = *s2++;
      if (c2 == L'\0')
      return c1 - c2;
    }
  while (c1 == c2);

  return c1 < c2 ? -1 : 1;
}

int
stupid_wcscmp (const wchar_t *s1, const wchar_t *s2)
{
  size_t ns1 = wcslen (s1) + 1;
  size_t ns2 = wcslen (s2) + 1;
  size_t n = ns1 < ns2 ? ns1 : ns2;
  int ret = 0;

  wchar_t c1, c2;

  while (n--) {
    c1 = *s1++;
    c2 = *s2++;
    if ((ret = c1 < c2 ? -1 : c1 == c2 ? 0 : 1) != 0)
      break;
  }
  return ret;
}

#else
# include <limits.h>

# define L(str) str
# define STRCMP strcmp
# define STRCPY strcpy
# define STRLEN strlen
# define MEMCPY memcpy
# define SIMPLE_STRCMP simple_strcmp
# define STUPID_STRCMP stupid_strcmp
# define CHAR char
# define UCHAR unsigned char
# define CHARBYTES 1
# define CHARBYTESLOG 0
# define CHARALIGN 1
# define MIDCHAR 0x7f
# define LARGECHAR 0xfe
# define CHAR__MAX CHAR_MAX
# define CHAR__MIN CHAR_MIN

/* Strcmp uses unsigned semantics for comparison. */
int
simple_strcmp (const char *s1, const char *s2)
{
  int ret;

  while ((ret = *(unsigned char *) s1 - *(unsigned char*) s2++) == 0 && *s1++);
  return ret;
}

int
stupid_strcmp (const char *s1, const char *s2)
{
  size_t ns1 = strlen (s1) + 1;
  size_t ns2 = strlen (s2) + 1;
  size_t n = ns1 < ns2 ? ns1 : ns2;
  int ret = 0;

  while (n--)
    if ((ret = *(unsigned char *) s1++ - *(unsigned char *) s2++) != 0)
      break;
  return ret;
}
#endif

typedef int (*proto_t) (const CHAR *, const CHAR *);

IMPL (STUPID_STRCMP, 1)
IMPL (SIMPLE_STRCMP, 1)
IMPL (STRCMP, 1)

static int
check_result (impl_t *impl,
	     const CHAR *s1, const CHAR *s2,
	     int exp_result)
{
  int result = CALL (impl, s1, s2);
  if ((exp_result == 0 && result != 0)
      || (exp_result < 0 && result >= 0)
      || (exp_result > 0 && result <= 0))
    {
      error (0, 0, "Wrong result in function %s %d %d", impl->name,
	     result, exp_result);
      ret = 1;
      return -1;
    }

  return 0;
}

static void
do_one_test (impl_t *impl,
	     const CHAR *s1, const CHAR *s2,
	     int exp_result)
{
  if (check_result (impl, s1, s2, exp_result) < 0)
    return;
}

static void
do_test (size_t align1, size_t align2, size_t len, int max_char,
	 int exp_result)
{
  size_t i;

  CHAR *s1, *s2;

  if (len == 0)
    return;

  align1 &= 63;
  if (align1 + (len + 1) * CHARBYTES >= page_size)
    return;

  align2 &= 63;
  if (align2 + (len + 1) * CHARBYTES >= page_size)
    return;

  /* Put them close to the end of page.  */
  i = align1 + CHARBYTES * (len + 2);
  s1 = (CHAR *) (buf1 + ((page_size - i) / 16 * 16) + align1);
  i = align2 + CHARBYTES * (len + 2);
  s2 = (CHAR *) (buf2 + ((page_size - i) / 16 * 16)  + align2);

  for (i = 0; i < len; i++)
    s1[i] = s2[i] = 1 + (23 << ((CHARBYTES - 1) * 8)) * i % max_char;

  s1[len] = s2[len] = 0;
  s1[len + 1] = 23;
  s2[len + 1] = 24 + exp_result;
  s2[len - 1] -= exp_result;

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s1, s2, exp_result);
}

static void
do_random_tests (void)
{
	UCHAR *p1 = (UCHAR *) (buf1 + page_size - 512 * CHARBYTES);
	UCHAR *p2 = (UCHAR *) (buf2 + page_size - 512 * CHARBYTES);

	for (size_t n = 0; n < ITERATIONS; n++)
	  {
	    /* for wcscmp case align1 and align2 mean here alignment
	       in wchar_t symbols, it equal 4*k alignment in bytes, we
	       don't check other alignments like for example
	       p1 = (wchar_t *)(buf1 + 1)
	       because it's wrong using of wchar_t type.  */
	    size_t align1 = random () & 31;
	    size_t align2;
	    if (random () & 1)
	      align2 = random () & 31;
	    else
	      align2 = align1 + (random () & 24);
	    size_t pos = random () & 511;
	    size_t j = align1 > align2 ? align1 : align2;
	    if (pos + j >= 511)
	      pos = 510 - j - (random () & 7);
	    size_t len1 = random () & 511;
	    if (pos >= len1 && (random () & 1))
	      len1 = pos + (random () & 7);
	    if (len1 + j >= 512)
	      len1 = 511 - j - (random () & 7);
	    size_t len2;
	    if (pos >= len1)
	      len2 = len1;
	    else
	      len2 = len1 + (len1 != 511 - j ? random () % (511 - j - len1) : 0);
	    j = (pos > len2 ? pos : len2) + align1 + 64;
	    if (j > 512)
	      j = 512;
	    for (size_t i = 0; i < j; ++i)
	      {
		p1[i] = random () & 255;
		if (i < len1 + align1 && !p1[i])
		  {
		    p1[i] = random () & 255;
		    if (!p1[i])
		      p1[i] = 1 + (random () & 127);
		  }
	      }
	    for (size_t i = 0; i < j; ++i)
	      {
		p2[i] = random () & 255;
		if (i < len2 + align2 && !p2[i])
		  {
		    p2[i] = random () & 255;
		    if (!p2[i])
		      p2[i] = 1 + (random () & 127);
		  }
	      }

	    int result = 0;
	    MEMCPY (p2 + align2, p1 + align1, pos);
	    if (pos < len1)
	      {
		if (p2[align2 + pos] == p1[align1 + pos])
		  {
		    p2[align2 + pos] = random () & 255;
		    if (p2[align2 + pos] == p1[align1 + pos])
		      p2[align2 + pos] = p1[align1 + pos] + 3 + (random () & 127);
		  }

		if (p1[align1 + pos] < p2[align2 + pos])
		  result = -1;
		else
		  result = 1;
	      }
	    p1[len1 + align1] = 0;
	    p2[len2 + align2] = 0;

	    FOR_EACH_IMPL (impl, 1)
	      {
		int r = CALL (impl, (CHAR *) (p1 + align1), (CHAR *) (p2 + align2));
		/* Test whether on 64-bit architectures where ABI requires
		   callee to promote has the promotion been done.  */
		asm ("" : "=g" (r) : "0" (r));
		if ((r == 0 && result)
		    || (r < 0 && result >= 0)
		    || (r > 0 && result <= 0))
		  {
		    error (0, 0, "Iteration %zd - wrong result in function %s (align in bytes: %zd, align in bytes: %zd, len1:  %zd, len2: %zd, pos: %zd) %d != %d, p1 %p p2 %p",
			   n, impl->name, (size_t) (p1 + align1) & 63, (size_t) (p1 + align2) & 63, len1, len2, pos, r, result, p1, p2);
		    ret = 1;
		  }
	      }
     }
}

static void
check (void)
{
  CHAR *s1 = (CHAR *) (buf1 + 0xb2c);
  CHAR *s2 = (CHAR *) (buf1 + 0xfd8);

  STRCPY(s1, L("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrs"));
  STRCPY(s2, L("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijkLMNOPQRSTUV"));

  /* Check correct working for negatives values */

  s1[0] = 1;
  s2[0] = 1;
  s1[1] = 1;
  s2[1] = 1;
  s1[2] = -1;
  s2[2] = 3;
  s1[3] = 0;
  s2[3] = -1;

  /* Check possible overflow bug, actual more for wcscmp */

  s1[7] = CHAR__MIN;
  s2[7] = CHAR__MAX;

  size_t l1 = STRLEN (s1);
  size_t l2 = STRLEN (s2);

  for (size_t i1 = 0; i1 < l1; i1++)
    for (size_t i2 = 0; i2 < l2; i2++)
      {
		int exp_result = SIMPLE_STRCMP (s1 + i1, s2 + i2);
		FOR_EACH_IMPL (impl, 0)
		check_result (impl, s1 + i1, s2 + i2, exp_result);
      }

  /* Test cases where there are multiple zero bytes after the first.  */

  for (size_t i = 0; i < 16 + 1; i++)
    {
      s1[i] = 0x00;
      s2[i] = 0x00;
    }

  for (size_t i = 0; i < 16; i++)
    {
      int exp_result;

      for (int val = 0x01; val < 0x100; val++)
	{
	  for (size_t j = 0; j < i; j++)
	    {
	      s1[j] = val;
	      s2[j] = val;
	    }

	  s2[i] = val;

	  exp_result = SIMPLE_STRCMP (s1, s2);
	  FOR_EACH_IMPL (impl, 0)
	    check_result (impl, s1, s2, exp_result);
	}
    }
}

static void
check2 (void)
{
  /* To trigger bug 25933, we need a size that is equal to the vector
     length times 4. In the case of AVX2 for Intel, we need 32 * 4.  We
     make this test generic and run it for all architectures as additional
     boundary testing for such related algorithms.  */
  size_t size = 32 * 4;
  CHAR *s1 = (CHAR *) (buf1 + (BUF1PAGES - 1) * page_size);
  CHAR *s2 = (CHAR *) (buf2 + (BUF1PAGES - 1) * page_size);
  int exp_result;

  memset (s1, 'a', page_size);
  memset (s2, 'a', page_size);
  s1[(page_size / CHARBYTES) - 1] = (CHAR) 0;
  s2[(page_size / CHARBYTES) - 1] = (CHAR) 0;

  /* Iterate over a size that is just below where we expect the bug to
     trigger up to the size we expect will trigger the bug e.g. [99-128].
     Likewise iterate the start of two strings between 30 and 31 bytes
     away from the boundary to simulate alignment changes.  */
  for (size_t s = 99; s <= size; s++)
    for (size_t s1a = 30; s1a < 32; s1a++)
      for (size_t s2a = 30; s2a < 32; s2a++)
	{
	  CHAR *s1p = s1 + (page_size / CHARBYTES - s) - s1a;
	  CHAR *s2p = s2 + (page_size / CHARBYTES - s) - s2a;
	  exp_result = SIMPLE_STRCMP (s1p, s2p);
	  FOR_EACH_IMPL (impl, 0)
	    check_result (impl, s1p, s2p, exp_result);
	}
}

int
test_main (void)
{
  size_t i;

  test_init ();
  check();
  check2 ();

  printf ("%23s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 1; i < 32; ++i)
    {
      do_test (CHARBYTES * i, CHARBYTES * i, i, MIDCHAR, 0);
      do_test (CHARBYTES * i, CHARBYTES * i, i, MIDCHAR, 1);
      do_test (CHARBYTES * i, CHARBYTES * i, i, MIDCHAR, -1);
    }

  for (i = 1; i < 10 + CHARBYTESLOG; ++i)
    {
      do_test (0, 0, 2 << i, MIDCHAR, 0);
      do_test (0, 0, 2 << i, LARGECHAR, 0);
      do_test (0, 0, 2 << i, MIDCHAR, 1);
      do_test (0, 0, 2 << i, LARGECHAR, 1);
      do_test (0, 0, 2 << i, MIDCHAR, -1);
      do_test (0, 0, 2 << i, LARGECHAR, -1);
      do_test (0, CHARBYTES * i, 2 << i, MIDCHAR, 1);
      do_test (CHARBYTES * i, CHARBYTES * (i + 1), 2 << i, LARGECHAR, 1);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (CHARBYTES * i, 2 * CHARBYTES * i, 8 << i, MIDCHAR, 0);
      do_test (2 * CHARBYTES * i, CHARBYTES * i, 8 << i, LARGECHAR, 0);
      do_test (CHARBYTES * i, 2 * CHARBYTES * i, 8 << i, MIDCHAR, 1);
      do_test (2 * CHARBYTES * i, CHARBYTES * i, 8 << i, LARGECHAR, 1);
      do_test (CHARBYTES * i, 2 * CHARBYTES * i, 8 << i, MIDCHAR, -1);
      do_test (2 * CHARBYTES * i, CHARBYTES * i, 8 << i, LARGECHAR, -1);
    }

  do_random_tests ();
  return ret;
}

#include <support/test-driver.c>
