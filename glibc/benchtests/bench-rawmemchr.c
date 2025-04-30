/* Measure memchr functions.
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

#include <assert.h>
#include <stdint.h>

#define TEST_MAIN
#define TEST_NAME "rawmemchr"
#include "bench-string.h"

#include "json-lib.h"

typedef char *(*proto_t) (const char *, int);

char *
generic_rawmemchr (const char *s, int c)
{
  if (c != 0)
    return memchr (s, c, PTRDIFF_MAX);
  return (char *)s + strlen (s);
}

IMPL (rawmemchr, 1)
IMPL (generic_rawmemchr, 0)

static void
do_one_test (json_ctx_t *json_ctx, impl_t *impl, const char *s, int c, char *exp_res)
{
  size_t i, iters = INNER_LOOP_ITERS_LARGE * 4;
  timing_t start, stop, cur;
  char *res = CALL (impl, s, c);
  if (res != exp_res)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     res, exp_res);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s, c);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  json_element_double (json_ctx, (double) cur / (double) iters);
}

static void
do_test (json_ctx_t *json_ctx, size_t align, size_t pos, size_t len, int seek_char)
{
  size_t i;
  char *result;

  align &= 7;
  if (align + len >= page_size)
    return;

  for (i = 0; i < len; ++i)
    {
      buf1[align + i] = 1 + 23 * i % 127;
      if (buf1[align + i] == seek_char)
	buf1[align + i] = seek_char + 1;
    }
  buf1[align + len] = 0;

  assert (pos < len);

  buf1[align + pos] = seek_char;
  buf1[align + len] = -seek_char;
  result = (char *) (buf1 + align + pos);

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "length", pos);
  json_attr_uint (json_ctx, "alignment", align);
  json_attr_uint (json_ctx, "char", seek_char);
  json_array_begin (json_ctx, "timings");

  FOR_EACH_IMPL (impl, 0)
    do_one_test (json_ctx, impl, (char *) (buf1 + align), seek_char, result);

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
  json_attr_string (&json_ctx, "bench-variant", "");

  json_array_begin (&json_ctx, "ifuncs");
  FOR_EACH_IMPL (impl, 0)
      json_element_string (&json_ctx, impl->name);
  json_array_end (&json_ctx);

  json_array_begin (&json_ctx, "results");

  for (i = 1; i < 7; ++i)
    {
      do_test (&json_ctx, 0, 16 << i, 2048, 23);
      do_test (&json_ctx, i, 64, 256, 23);
      do_test (&json_ctx, 0, 16 << i, 2048, 0);
      do_test (&json_ctx, i, 64, 256, 0);
    }
  for (i = 1; i < 32; ++i)
    {
      do_test (&json_ctx, 0, i, i + 1, 23);
      do_test (&json_ctx, 0, i, i + 1, 0);
    }

  json_array_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_document_end (&json_ctx);

  return ret;
}

#include <support/test-driver.c>
