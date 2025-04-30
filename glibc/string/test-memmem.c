/* Test and measure memmem functions.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Ulrich Drepper <drepper@redhat.com>, 2008.

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
#define TEST_NAME "memmem"
#define BUF1PAGES 20
#define ITERATIONS 500
#include "test-string.h"

typedef char *(*proto_t) (const void *, size_t, const void *, size_t);
void *simple_memmem (const void *, size_t, const void *, size_t);

IMPL (simple_memmem, 0)
IMPL (memmem, 1)

void *
simple_memmem (const void *haystack, size_t haystack_len, const void *needle,
	       size_t needle_len)
{
  const char *begin;
  const char *const last_possible
    = (const char *) haystack + haystack_len - needle_len;

  if (needle_len == 0)
    /* The first occurrence of the empty string is deemed to occur at
       the beginning of the string.  */
    return (void *) haystack;

  /* Sanity check, otherwise the loop might search through the whole
     memory.  */
  if (__glibc_unlikely (haystack_len < needle_len))
    return NULL;

  for (begin = (const char *) haystack; begin <= last_possible; ++begin)
    if (begin[0] == ((const char *) needle)[0]
        && !memcmp ((const void *) &begin[1],
		    (const void *) ((const char *) needle + 1),
		    needle_len - 1))
      return (void *) begin;

  return NULL;
}

static int
check_result (impl_t *impl, const void *haystack, size_t haystack_len,
	     const void *needle, size_t needle_len, const void *expected)
{
  void *res;

  res = CALL (impl, haystack, haystack_len, needle, needle_len);
  if (res != expected)
    {
      error (0, 0, "Wrong result in function %s %p %p", impl->name,
	     res, expected);
      ret = 1;
      return -1;
    }

  return 0;
}

static void
do_one_test (impl_t *impl, const void *haystack, size_t haystack_len,
	     const void *needle, size_t needle_len, const void *expected)
{
  if (check_result (impl, haystack, haystack_len, needle, needle_len,
		    expected) < 0)
    return;
}

static void
do_test (const char *str, size_t len, size_t idx)
{
  char tmpbuf[len];

  memcpy (tmpbuf, buf1 + idx, len);
  memcpy (buf1 + idx, str, len);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, buf1, BUF1PAGES * page_size, str, len, buf1 + idx);

  memcpy (buf1 + idx, tmpbuf, len);
}

static void
do_random_tests (void)
{
  for (size_t n = 0; n < ITERATIONS; ++n)
    {
      char tmpbuf[32];

      size_t shift = random () % 11;
      size_t rel = random () % ((2 << (shift + 1)) * 64);
      size_t idx = MIN ((2 << shift) * 64 + rel, BUF1PAGES * page_size - 2);
      size_t len = random () % (sizeof (tmpbuf) - 1) + 1;
      len = MIN (len, BUF1PAGES * page_size - idx - 1);
      memcpy (tmpbuf, buf1 + idx, len);
      for (size_t i = random () % len / 2 + 1; i > 0; --i)
	{
	  size_t off = random () % len;
	  char ch = '0' + random () % 10;

	  buf1[idx + off] = ch;
	}

      FOR_EACH_IMPL (impl, 0)
	do_one_test (impl, buf1, BUF1PAGES * page_size, buf1 + idx, len,
		     buf1 + idx);

      memcpy (buf1 + idx, tmpbuf, len);
    }
}

static void
check1 (void)
{

  const char search_buf_data[5] = { 0x56, 0x34, 0x12, 0x78, 0x78 };
  const char pattern[2] = { 0x78, 0x56 };
  void *search_buf = (void *) buf1 + page_size - sizeof search_buf_data;
  void *exp_result;

  memcpy (search_buf, search_buf_data, sizeof search_buf_data);
  exp_result = simple_memmem (search_buf, sizeof search_buf_data,
			      pattern, sizeof pattern);
  FOR_EACH_IMPL (impl, 0)
    check_result (impl, search_buf, sizeof search_buf_data,
		  pattern, sizeof pattern, exp_result);
}

static const char *const strs[] =
  {
    "00000", "00112233", "0123456789", "0000111100001111",
    "00000111110000022222", "012345678901234567890",
    "abc0", "aaaa0", "abcabc0"
  };


int
test_main (void)
{
  size_t i;

  test_init ();

  check1 ();

  printf ("%23s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 0; i < BUF1PAGES * page_size; ++i)
    buf1[i] = 60 + random () % 32;

  for (i = 0; i < sizeof (strs) / sizeof (strs[0]); ++i)
    for (size_t j = 0; j < 120; j += 7)
      {
	size_t len = strlen (strs[i]);

	do_test (strs[i], len, j);
      }

  do_random_tests ();
  return ret;
}

#include <support/test-driver.c>
