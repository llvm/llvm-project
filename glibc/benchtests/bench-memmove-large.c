/* Measure memmove functions with large data sizes.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#define BASE_PAGE_SIZE (1024 * 1024)
#define START_SIZE (4 * 1024)
#define MIN_PAGE_SIZE (getpagesize () + 16 * 1024 * 1024)
#define TEST_MAIN
#define TEST_NAME "memmove"
#define TIMEOUT (20 * 60)
#include "bench-string.h"
#include "json-lib.h"

IMPL (memmove, 1)

typedef char *(*proto_t) (char *, const char *, size_t);

static void
do_one_test (json_ctx_t *json_ctx, impl_t *impl, char *dst, char *src,
	     size_t len)
{
  size_t i, iters = 16;
  timing_t start, stop, cur;

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, dst, src, len);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  json_element_double (json_ctx, (double) cur / (double) iters);
}

static void
do_test (json_ctx_t *json_ctx, size_t align1, size_t align2, size_t len)
{
  size_t i, j;
  char *s1, *s2;

  align1 &= 127;
  if (align1 + len >= page_size)
    return;

  align2 &= 127;
  if (align2 + len >= page_size)
    return;

  s1 = (char *) (buf2 + align1);
  s2 = (char *) (buf2 + align2);

  for (i = 0, j = 1; i < len; i++, j += 23)
    s1[i] = j;

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "length", (double) len);
  json_attr_uint (json_ctx, "align1", (double) align1);
  json_attr_uint (json_ctx, "align2", (double) align2);
  json_array_begin (json_ctx, "timings");

  FOR_EACH_IMPL (impl, 0)
    do_one_test (json_ctx, impl, s2, s1, len);

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
  json_attr_object_begin (&json_ctx, "memmove");
  json_attr_string (&json_ctx, "bench-variant", "large");

  json_array_begin (&json_ctx, "ifuncs");

  FOR_EACH_IMPL (impl, 0)
    json_element_string (&json_ctx, impl->name);
  json_array_end (&json_ctx);

  json_array_begin (&json_ctx, "results");
  for (i = START_SIZE; i <= MIN_PAGE_SIZE; i <<= 1)
    {
      do_test (&json_ctx, 0, 64, i + 7);
      do_test (&json_ctx, 0, 3, i + 15);
      do_test (&json_ctx, 3, 0, i + 31);
      do_test (&json_ctx, 3, 7, i + 63);
      do_test (&json_ctx, 9, 5, i + 127);
    }

  json_array_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_document_end (&json_ctx);

  return ret;
}

#include <support/test-driver.c>
