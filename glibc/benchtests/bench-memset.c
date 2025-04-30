/* Measure memset functions.
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
#ifndef WIDE
# define TEST_NAME "memset"
#else
# define TEST_NAME "wmemset"
# define generic_memset generic_wmemset
#endif /* WIDE */
#define MIN_PAGE_SIZE 131072
#include "bench-string.h"

#include "json-lib.h"

#ifdef WIDE
CHAR *generic_wmemset (CHAR *, CHAR, size_t);
#else
void *generic_memset (void *, int, size_t);
#endif

typedef void *(*proto_t) (void *, int, size_t);

IMPL (MEMSET, 1)
IMPL (generic_memset, 0)

static void
do_one_test (json_ctx_t *json_ctx, impl_t *impl, CHAR *s,
	     int c __attribute ((unused)), size_t n)
{
  size_t i, iters = INNER_LOOP_ITERS;
  timing_t start, stop, cur;

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s, c, n);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  json_element_double (json_ctx, (double) cur / (double) iters);
}

static void
do_test (json_ctx_t *json_ctx, size_t align, int c, size_t len)
{
  align &= 4095;
  if ((align + len) * sizeof (CHAR) > page_size)
    return;

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "length", len);
  json_attr_uint (json_ctx, "alignment", align);
  json_attr_int (json_ctx, "char", c);
  json_array_begin (json_ctx, "timings");

  FOR_EACH_IMPL (impl, 0)
    {
      do_one_test (json_ctx, impl, (CHAR *) (buf1) + align, c, len);
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
  int c = 0;

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

  for (c = -65; c <= 130; c += 65)
    {
      for (i = 0; i < 18; ++i)
	do_test (&json_ctx, 0, c, 1 << i);
      for (i = 1; i < 64; ++i)
	{
	  do_test (&json_ctx, i, c, i);
	  do_test (&json_ctx, 4096 - i, c, i);
	  do_test (&json_ctx, 4095, c, i);
	  if (i & (i - 1))
	    do_test (&json_ctx, 0, c, i);
	}
      for (i = 32; i < 512; i+=32)
	{
	  do_test (&json_ctx, 0, c, i);
	  do_test (&json_ctx, i, c, i);
	}
      do_test (&json_ctx, 1, c, 14);
      do_test (&json_ctx, 3, c, 1024);
      do_test (&json_ctx, 4, c, 64);
      do_test (&json_ctx, 2, c, 25);
    }
  for (i = 33; i <= 256; i += 4)
    {
      do_test (&json_ctx, 0, c, 32 * i);
      do_test (&json_ctx, i, c, 32 * i);
    }

  json_array_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_document_end (&json_ctx);

  return ret;
}

#include <support/test-driver.c>

#define libc_hidden_builtin_def(X)
#define libc_hidden_def(X)
#define libc_hidden_weak(X)
#define weak_alias(X,Y)
#ifndef WIDE
# undef MEMSET
# define MEMSET generic_memset
# include <string/memset.c>
#else
# define WMEMSET generic_wmemset
# include <wcsmbs/wmemset.c>
#endif
