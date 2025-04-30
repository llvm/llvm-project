/* Measure memcmp functions.
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
# define TEST_NAME "wmemcmp"
#else
# define TEST_NAME "memcmp"
#endif
#include "bench-string.h"
#ifdef WIDE

# define SIMPLE_MEMCMP simple_wmemcmp
int
simple_wmemcmp (const wchar_t *s1, const wchar_t *s2, size_t n)
{
  int ret = 0;
  /* Warning!
	wmemcmp has to use SIGNED comparison for elements.
	memcmp has to use UNSIGNED comparison for elemnts.
  */
  while (n-- && (ret = *s1 < *s2 ? -1 : *s1 == *s2 ? 0 : 1) == 0) {s1++; s2++;}
  return ret;
}
#else
# include <limits.h>

# define SIMPLE_MEMCMP simple_memcmp

int
simple_memcmp (const char *s1, const char *s2, size_t n)
{
  int ret = 0;

  while (n-- && (ret = *(unsigned char *) s1++ - *(unsigned char *) s2++) == 0);
  return ret;
}
#endif

# include "json-lib.h"

typedef int (*proto_t) (const CHAR *, const CHAR *, size_t);

IMPL (SIMPLE_MEMCMP, 0)
IMPL (MEMCMP, 1)

static void
do_one_test (json_ctx_t *json_ctx, impl_t *impl, const CHAR *s1,
	     const CHAR *s2, size_t len, int exp_result)
{
  size_t i, iters = INNER_LOOP_ITERS8;
  timing_t start, stop, cur;

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s1, s2, len);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  json_element_double (json_ctx, (double) cur / (double) iters);
}

static void
do_test (json_ctx_t *json_ctx, size_t align1, size_t align2, size_t len,
	 int exp_result)
{
  size_t i;
  CHAR *s1, *s2;

  if (len == 0)
    return;

  align1 &= (4096 - CHARBYTES);
  if (align1 + (len + 1) * CHARBYTES >= page_size)
    return;

  align2 &= (4096 - CHARBYTES);
  if (align2 + (len + 1) * CHARBYTES >= page_size)
    return;

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "length", (double) len);
  json_attr_uint (json_ctx, "align1", (double) align1);
  json_attr_uint (json_ctx, "align2", (double) align2);
  json_attr_uint (json_ctx, "result", (double) exp_result);
  json_array_begin (json_ctx, "timings");

  FOR_EACH_IMPL (impl, 0)
    {
      s1 = (CHAR *) (buf1 + align1);
      s2 = (CHAR *) (buf2 + align2);

      for (i = 0; i < len; i++)
	s1[i] = s2[i] = 1 + (23 << ((CHARBYTES - 1) * 8)) * i % MAX_CHAR;

      s1[len] = align1;
      s2[len] = align2;
      s2[len - 1] -= exp_result;

      do_one_test (json_ctx, impl, s1, s2, len, exp_result);
      alloc_bufs ();
    }

  json_array_end (json_ctx);
  json_element_object_end (json_ctx);
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
  for (i = 1; i < 32; ++i)
    {
      do_test (&json_ctx, i * CHARBYTES, i * CHARBYTES, i, 0);
      do_test (&json_ctx, i * CHARBYTES, i * CHARBYTES, i, 1);
      do_test (&json_ctx, i * CHARBYTES, i * CHARBYTES, i, -1);
    }

  for (i = 0; i < 32; ++i)
    {
      do_test (&json_ctx, 0, 0, i, 0);
      do_test (&json_ctx, 0, 0, i, 1);
      do_test (&json_ctx, 0, 0, i, -1);
      do_test (&json_ctx, 4096 - i, 0, i, 0);
      do_test (&json_ctx, 4096 - i, 0, i, 1);
      do_test (&json_ctx, 4096 - i, 0, i, -1);
    }

  for (i = 33; i < 385; i += 32)
    {
      do_test (&json_ctx, 0, 0, i, 0);
      do_test (&json_ctx, 0, 0, i, 1);
      do_test (&json_ctx, 0, 0, i, -1);
      do_test (&json_ctx, i, 0, i, 0);
      do_test (&json_ctx, 0, i, i, 1);
      do_test (&json_ctx, i, i, i, -1);
    }

  for (i = 1; i < 10; ++i)
    {
      do_test (&json_ctx, 0, 0, 2 << i, 0);
      do_test (&json_ctx, 0, 0, 2 << i, 1);
      do_test (&json_ctx, 0, 0, 2 << i, -1);
      do_test (&json_ctx, (8 - i) * CHARBYTES, (2 * i) * CHARBYTES, 16 << i, 0);
      do_test (&json_ctx, 0, 0, 16 << i, 0);
      do_test (&json_ctx, 0, 0, 16 << i, 1);
      do_test (&json_ctx, 0, 0, 16 << i, -1);
      do_test (&json_ctx, i, 0, 2 << i, 0);
      do_test (&json_ctx, 0, i, 2 << i, 1);
      do_test (&json_ctx, i, i, 2 << i, -1);
      do_test (&json_ctx, i, 0, 16 << i, 0);
      do_test (&json_ctx, 0, i, 16 << i, 1);
      do_test (&json_ctx, i, i, 16 << i, -1);
    }

  for (i = 1; i < 10; ++i)
    {
      do_test (&json_ctx, i * CHARBYTES, 2 * (i * CHARBYTES), 8 << i, 0);
      do_test (&json_ctx, i * CHARBYTES, 2 * (i * CHARBYTES), 8 << i, 1);
      do_test (&json_ctx, i * CHARBYTES, 2 * (i * CHARBYTES), 8 << i, -1);
    }

  json_array_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_document_end (&json_ctx);

  return ret;
}

#include <support/test-driver.c>
