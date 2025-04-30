/* Measure STRLEN functions.
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
# define TEST_NAME "strlen"
#else
# define TEST_NAME "wcslen"
# define generic_strlen generic_wcslen
# define memchr_strlen wcschr_wcslen
#endif
#include "bench-string.h"

#include "json-lib.h"

typedef size_t (*proto_t) (const CHAR *);

size_t generic_strlen (const CHAR *);
size_t memchr_strlen (const CHAR *);

IMPL (memchr_strlen, 0)
IMPL (generic_strlen, 0)

size_t
memchr_strlen (const CHAR *p)
{
  return (const CHAR *)MEMCHR (p, 0, PTRDIFF_MAX) - p;
}

IMPL (STRLEN, 1)


static void
do_one_test (json_ctx_t *json_ctx, impl_t *impl, const CHAR *s, size_t exp_len)
{
  size_t len = CALL (impl, s), i, iters = INNER_LOOP_ITERS_LARGE;
  timing_t start, stop, cur;

  if (len != exp_len)
    {
      error (0, 0, "Wrong result in function %s %zd %zd", impl->name,
	     len, exp_len);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, s);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  json_element_double (json_ctx, (double) cur / (double) iters);
}

static void
do_test (json_ctx_t *json_ctx, size_t align, size_t len)
{
  size_t i;

  align &= 63;
  if (align + sizeof (CHAR) * len >= page_size)
    return;

  json_element_object_begin (json_ctx);
  json_attr_uint (json_ctx, "length", len);
  json_attr_uint (json_ctx, "alignment", align);
  json_array_begin (json_ctx, "timings");


  FOR_EACH_IMPL (impl, 0)
    {
      CHAR *buf = (CHAR *) (buf1);

      for (i = 0; i < len; ++i)
	buf[align + i] = 1 + 11111 * i % MAX_CHAR;
      buf[align + len] = 0;

      do_one_test (json_ctx, impl, (CHAR *) (buf + align), len);
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
  json_attr_string (&json_ctx, "bench-variant", "");

  json_array_begin (&json_ctx, "ifuncs");
  FOR_EACH_IMPL (impl, 0)
    json_element_string (&json_ctx, impl->name);
  json_array_end (&json_ctx);

  json_array_begin (&json_ctx, "results");
  /* Checking with only 4 * N alignments for wcslen, other alignments are wrong for wchar_t type arrays*/

  for (i = 1; i < 8; ++i)
  {
    do_test (&json_ctx, sizeof (CHAR) * i, i);
    do_test (&json_ctx, 0, i);
  }

  for (i = 2; i <= 12; ++i)
    {
      do_test (&json_ctx, 0, 1 << i);
      do_test (&json_ctx, sizeof (CHAR) * 7, 1 << i);
      do_test (&json_ctx, sizeof (CHAR) * i, 1 << i);
      do_test (&json_ctx, sizeof (CHAR) * i, (size_t)((1 << i) / 1.5));
    }

  json_array_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_attr_object_end (&json_ctx);
  json_document_end (&json_ctx);

  return ret;
}

#include <support/test-driver.c>

#define libc_hidden_builtin_def(X)
#ifndef WIDE
# undef STRLEN
# define STRLEN generic_strlen
# include <string/strlen.c>
#else
# define WCSLEN generic_strlen
# include <wcsmbs/wcslen.c>
#endif
