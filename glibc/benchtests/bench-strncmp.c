/* Measure strncmp functions.
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
#ifdef WIDE
# define TEST_NAME "wcsncmp"
#else
# define TEST_NAME "strncmp"
#endif /* !WIDE */
#include "bench-string.h"
#include "json-lib.h"

#ifdef WIDE
# define L(str) L##str
# define STRDUP wcsdup
# define SIMPLE_STRNCMP simple_wcsncmp

/* Wcsncmp uses signed semantics for comparison, not unsigned.
   Avoid using substraction since possible overflow.  */
int
simple_wcsncmp (const CHAR *s1, const CHAR *s2, size_t n)
{
  wchar_t c1, c2;

  while (n--)
    {
      c1 = *s1++;
      c2 = *s2++;
      if (c1 == L ('\0') || c1 != c2)
	return c1 > c2 ? 1 : (c1 < c2 ? -1 : 0);
    }
  return 0;
}

#else
# define L(str) str
# define STRDUP strdup
# define SIMPLE_STRNCMP simple_strncmp

/* Strncmp uses unsigned semantics for comparison.  */
int
simple_strncmp (const char *s1, const char *s2, size_t n)
{
  int ret = 0;

  while (n-- && (ret = *(unsigned char *) s1 - * (unsigned char *) s2++) == 0
	 && *s1++);
  return ret;
}

#endif /* !WIDE */

typedef int (*proto_t) (const CHAR *, const CHAR *, size_t);

IMPL (SIMPLE_STRNCMP, 0)
IMPL (STRNCMP, 1)


static void
do_one_test (json_ctx_t *json_ctx, impl_t *impl, const CHAR *s1, const CHAR
	     *s2, size_t n, int exp_result)
{
  size_t i, iters = INNER_LOOP_ITERS8;
  timing_t start, stop, cur;

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s1, s2, n);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  json_element_double (json_ctx, (double) cur / (double) iters);
}

static void
do_test_limit (json_ctx_t *json_ctx, size_t align1, size_t align2, size_t len,
	       size_t n, int max_char, int exp_result)
{
  size_t i, align_n;
  CHAR *s1, *s2;

  align1 &= 15;
  align2 &= 15;
  align_n = (page_size - n * CHARBYTES) & 15;

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "strlen", (double) len);
  json_attr_uint (json_ctx, "len", (double) n);
  json_attr_uint (json_ctx, "align1", (double) align1);
  json_attr_uint (json_ctx, "align2", (double) align2);
  json_array_begin (json_ctx, "timings");

  FOR_EACH_IMPL (impl, 0)
    {
      alloc_bufs ();
      s1 = (CHAR *) (buf1 + page_size - n * CHARBYTES);
      s2 = (CHAR *) (buf2 + page_size - n * CHARBYTES);

      if (align1 < align_n)
	s1 = (CHAR *) ((char *) s1 - (align_n - align1));

      if (align2 < align_n)
	s2 = (CHAR *) ((char *) s2 - (align_n - align2));

      for (i = 0; i < n; i++)
	s1[i] = s2[i] = 1 + 23 * i % max_char;

      if (len < n)
	{
	  s1[len] = 0;
	  s2[len] = 0;
	  if (exp_result < 0)
	    s2[len] = 32;
	  else if (exp_result > 0)
	    s1[len] = 64;
	}

      do_one_test (json_ctx, impl, s1, s2, n, exp_result);
    }

  json_array_end (json_ctx);
  json_element_object_end (json_ctx);
}

static void
do_test (json_ctx_t *json_ctx, size_t align1, size_t align2, size_t len, size_t
	 n, int max_char, int exp_result)
{
  size_t i;
  CHAR *s1, *s2;

  if (n == 0)
    return;

  align1 &= 63;
  if (align1 + (n + 1) * CHARBYTES >= page_size)
    return;

  align2 &= 7;
  if (align2 + (n + 1) * CHARBYTES >= page_size)
    return;

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "strlen", (double) len);
  json_attr_uint (json_ctx, "len", (double) n);
  json_attr_uint (json_ctx, "align1", (double) align1);
  json_attr_uint (json_ctx, "align2", (double) align2);
  json_array_begin (json_ctx, "timings");

  FOR_EACH_IMPL (impl, 0)
    {
      alloc_bufs ();
      s1 = (CHAR *) (buf1 + align1);
      s2 = (CHAR *) (buf2 + align2);

      for (i = 0; i < n; i++)
	s1[i] = s2[i] = 1 + (23 << ((CHARBYTES - 1) * 8)) * i % max_char;

      s1[n] = 24 + exp_result;
      s2[n] = 23;
      s1[len] = 0;
      s2[len] = 0;
      if (exp_result < 0)
	s2[len] = 32;
      else if (exp_result > 0)
	s1[len] = 64;
      if (len >= n)
	s2[n - 1] -= exp_result;

      do_one_test (json_ctx, impl, s1, s2, n, exp_result);
    }

  json_array_end (json_ctx);
  json_element_object_end (json_ctx);
}

static void
do_one_test_page_boundary (json_ctx_t *json_ctx, CHAR *s1, CHAR *s2,
			   size_t align1, size_t align2, size_t len,
			   size_t n, int exp_result)
{
  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "strlen", (double) len);
  json_attr_uint (json_ctx, "len", (double) n);
  json_attr_uint (json_ctx, "align1", (double) align1);
  json_attr_uint (json_ctx, "align2", (double) align2);
  json_array_begin (json_ctx, "timings");
  FOR_EACH_IMPL (impl, 0)
    do_one_test (json_ctx, impl, s1, s2, n, exp_result);
  json_array_end (json_ctx);
  json_element_object_end (json_ctx);
}

static void
do_test_page_boundary (json_ctx_t *json_ctx)
{
  /* To trigger bug 25933, we need a size that is equal to the vector
     length times 4. In the case of AVX2 for Intel, we need 32 * 4.  We
     make this test generic and run it for all architectures as additional
     boundary testing for such related algorithms.  */
  size_t size = 32 * 4;
  size_t len;
  CHAR *s1 = (CHAR *) (buf1 + (BUF1PAGES - 1) * page_size);
  CHAR *s2 = (CHAR *) (buf2 + (BUF1PAGES - 1) * page_size);
  int exp_result;

  memset (s1, 'a', page_size);
  memset (s2, 'a', page_size);

  s1[(page_size / CHARBYTES) - 1] = (CHAR) 0;

  /* Iterate over a size that is just below where we expect the bug to
     trigger up to the size we expect will trigger the bug e.g. [99-128].
     Likewise iterate the start of two strings between 30 and 31 bytes
     away from the boundary to simulate alignment changes.  */
  for (size_t s = 99; s <= size; s++)
    for (size_t s1a = 30; s1a < 32; s1a++)
      for (size_t s2a = 30; s2a < 32; s2a++)
	{
	  size_t align1 = (page_size / CHARBYTES - s) - s1a;
	  size_t align2 = (page_size / CHARBYTES - s) - s2a;
	  CHAR *s1p = s1 + align1;
	  CHAR *s2p = s2 + align2;
	  len = (page_size / CHARBYTES) - 1 - align1;
	  exp_result = SIMPLE_STRNCMP (s1p, s2p, s);
	  do_one_test_page_boundary (json_ctx, s1p, s2p, align1, align2,
				     len, s, exp_result);
	}
}

static void
do_one_test_page (json_ctx_t *json_ctx, size_t offset1, size_t offset2,
		  CHAR *s2)
{
  CHAR *s1;
  int exp_result;

  if (offset1 * CHARBYTES  >= page_size
      || offset2 * CHARBYTES >= page_size)
    return;

  s1 = (CHAR *) buf1;
  s1 += offset1;
  s2 += offset2;

  size_t len = (page_size / CHARBYTES) - offset1;

  exp_result= *s1;

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "strlen", (double) len);
  json_attr_uint (json_ctx, "len", (double) page_size);
  json_attr_uint (json_ctx, "align1", (double) offset1);
  json_attr_uint (json_ctx, "align2", (double) offset2);
  json_array_begin (json_ctx, "timings");
  {
    FOR_EACH_IMPL (impl, 0)
      do_one_test (json_ctx, impl, s1, s2, page_size, -exp_result);
  }
  json_array_end (json_ctx);
  json_element_object_end (json_ctx);

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "strlen", (double) len);
  json_attr_uint (json_ctx, "len", (double) page_size);
  json_attr_uint (json_ctx, "align1", (double) offset1);
  json_attr_uint (json_ctx, "align2", (double) offset2);
  json_array_begin (json_ctx, "timings");
  {
    FOR_EACH_IMPL (impl, 0)
      do_one_test (json_ctx, impl, s1, s2, page_size, exp_result);
  }
  json_array_end (json_ctx);
  json_element_object_end (json_ctx);
}

static void
do_test_page (json_ctx_t *json_ctx)
{
  size_t i;
  CHAR *s1, *s2;

  s1 = (CHAR *) buf1;
  /* Fill buf1 with 23. */
  for (i = 0; i < (page_size / CHARBYTES) - 1; i++)
    s1[i] = 23;
  s1[i] = 0;

  /* Make a copy of buf1.  */
  s2 = STRDUP (s1);

  /* Test should terminate within the page boundary.  */
  for (i = 0; i < (108 / CHARBYTES); ++i)
    do_one_test_page (json_ctx, ((page_size - 108) / CHARBYTES) + i,
		      ((page_size - 1460) / CHARBYTES), s2);

  free (s2);
}

int
test_main (void)
{
  json_ctx_t json_ctx;
  size_t i;

  test_init ();

  json_init (&json_ctx, 0, stdout);

  json_document_begin (&json_ctx);
  json_attr_string (&json_ctx, "timing_type", TIMING_TYPE);

  json_attr_object_begin (&json_ctx, "functions");
  json_attr_object_begin (&json_ctx, TEST_NAME);
  json_attr_string (&json_ctx, "bench-variant", "default");

  json_array_begin (&json_ctx, "ifuncs");
  FOR_EACH_IMPL (impl, 0)
    json_element_string (&json_ctx, impl->name);
  json_array_end (&json_ctx);

  json_array_begin (&json_ctx, "results");

  for (i =0; i < 16; ++i)
    {
      do_test (&json_ctx, 0, 0, 8, i, 127, 0);
      do_test (&json_ctx, 0, 0, 8, i, 127, -1);
      do_test (&json_ctx, 0, 0, 8, i, 127, 1);
      do_test (&json_ctx, i, i, 8, i, 127, 0);
      do_test (&json_ctx, i, i, 8, i, 127, 1);
      do_test (&json_ctx, i, i, 8, i, 127, -1);
      do_test (&json_ctx, i, 2 * i, 8, i, 127, 0);
      do_test (&json_ctx, 2 * i, i, 8, i, 127, 1);
      do_test (&json_ctx, i, 3 * i, 8, i, 127, -1);
      do_test (&json_ctx, 0, 0, 8, i, 255, 0);
      do_test (&json_ctx, 0, 0, 8, i, 255, -1);
      do_test (&json_ctx, 0, 0, 8, i, 255, 1);
      do_test (&json_ctx, i, i, 8, i, 255, 0);
      do_test (&json_ctx, i, i, 8, i, 255, 1);
      do_test (&json_ctx, i, i, 8, i, 255, -1);
      do_test (&json_ctx, i, 2 * i, 8, i, 255, 0);
      do_test (&json_ctx, 2 * i, i, 8, i, 255, 1);
      do_test (&json_ctx, i, 3 * i, 8, i, 255, -1);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (&json_ctx, 0, 0, 8 << i, 16 << i, 127, 0);
      do_test (&json_ctx, 0, 0, 8 << i, 16 << i, 127, 1);
      do_test (&json_ctx, 0, 0, 8 << i, 16 << i, 127, -1);
      do_test (&json_ctx, 0, 0, 8 << i, 16 << i, 255, 0);
      do_test (&json_ctx, 0, 0, 8 << i, 16 << i, 255, 1);
      do_test (&json_ctx, 0, 0, 8 << i, 16 << i, 255, -1);
      do_test (&json_ctx, 8 - i, 2 * i, 8 << i, 16 << i, 127, 0);
      do_test (&json_ctx, 8 - i, 2 * i, 8 << i, 16 << i, 127, 1);
      do_test (&json_ctx, 2 * i, i, 8 << i, 16 << i, 255, 0);
      do_test (&json_ctx, 2 * i, i, 8 << i, 16 << i, 255, 1);
    }

  do_test_limit (&json_ctx, 4, 0, 21, 20, 127, 0);
  do_test_limit (&json_ctx, 0, 4, 21, 20, 127, 0);
  do_test_limit (&json_ctx, 8, 0, 25, 24, 127, 0);
  do_test_limit (&json_ctx, 0, 8, 25, 24, 127, 0);

  for (i = 0; i < 8; ++i)
    {
      do_test_limit (&json_ctx, 0, 0, 17 - i, 16 - i, 127, 0);
      do_test_limit (&json_ctx, 0, 0, 17 - i, 16 - i, 255, 0);
      do_test_limit (&json_ctx, 0, 0, 15 - i, 16 - i, 127, 0);
      do_test_limit (&json_ctx, 0, 0, 15 - i, 16 - i, 127, 1);
      do_test_limit (&json_ctx, 0, 0, 15 - i, 16 - i, 127, -1);
      do_test_limit (&json_ctx, 0, 0, 15 - i, 16 - i, 255, 0);
      do_test_limit (&json_ctx, 0, 0, 15 - i, 16 - i, 255, 1);
      do_test_limit (&json_ctx, 0, 0, 15 - i, 16 - i, 255, -1);
    }

  do_test_page_boundary (&json_ctx);
  do_test_page (&json_ctx);

  json_array_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_document_end (&json_ctx);

  return ret;
}

#include <support/test-driver.c>
