/* Test and measure strstr functions.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Ulrich Drepper <drepper@redhat.com>, 2010.

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
#define TEST_NAME "strstr"
#include "test-string.h"


#define STRSTR simple_strstr
#define libc_hidden_builtin_def(arg) /* nothing */
#define __strnlen strnlen
#include "strstr.c"


static char *
stupid_strstr (const char *s1, const char *s2)
{
  ssize_t s1len = strlen (s1);
  ssize_t s2len = strlen (s2);

  if (s2len > s1len)
    return NULL;

  for (ssize_t i = 0; i <= s1len - s2len; ++i)
    {
      size_t j;
      for (j = 0; j < s2len; ++j)
	if (s1[i + j] != s2[j])
	  break;
      if (j == s2len)
	return (char *) s1 + i;
    }

  return NULL;
}


typedef char *(*proto_t) (const char *, const char *);

IMPL (stupid_strstr, 0)
IMPL (simple_strstr, 0)
IMPL (strstr, 1)


static int
check_result (impl_t *impl, const char *s1, const char *s2,
	      char *exp_result)
{
  char *result = CALL (impl, s1, s2);
  if (result != exp_result)
    {
      error (0, 0, "Wrong result in function %s %s %s", impl->name,
	     (result == NULL) ? "(null)" : result,
	     (exp_result == NULL) ? "(null)" : exp_result);
      ret = 1;
      return -1;
    }

  return 0;
}

static void
do_one_test (impl_t *impl, const char *s1, const char *s2, char *exp_result)
{
  if (check_result (impl, s1, s2, exp_result) < 0)
    return;
}


static void
do_test (size_t align1, size_t align2, size_t len1, size_t len2,
	 int fail)
{
  char *s1 = (char *) (buf1 + align1);
  char *s2 = (char *) (buf2 + align2);

  static const char d[] = "1234567890abcdef";
#define dl (sizeof (d) - 1)
  char *ss2 = s2;
  for (size_t l = len2; l > 0; l = l > dl ? l - dl : 0)
    {
      size_t t = l > dl ? dl : l;
      ss2 = mempcpy (ss2, d, t);
    }
  s2[len2] = '\0';

  if (fail)
    {
      char *ss1 = s1;
      for (size_t l = len1; l > 0; l = l > dl ? l - dl : 0)
	{
	  size_t t = l > dl ? dl : l;
	  memcpy (ss1, d, t);
	  ++ss1[len2 > 7 ? 7 : len2 - 1];
	  ss1 += t;
	}
    }
  else
    {
      memset (s1, '0', len1);
      memcpy (s1 + len1 - len2, s2, len2);
    }
  s1[len1] = '\0';

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s1, s2, fail ? NULL : s1 + len1 - len2);

}

static void
check1 (void)
{
  const char s1[] =
    "F_BD_CE_BD_EF_BF_BD_EF_BF_BD_EF_BF_BD_EF_BF_BD_C3_88_20_EF_BF_BD_EF_BF_BD_EF_BF_BD_C3_A7_20_EF_BF_BD";
  const char s2[] = "_EF_BF_BD_EF_BF_BD_EF_BF_BD_EF_BF_BD_EF_BF_BD";
  char *exp_result;

  exp_result = stupid_strstr (s1, s2);
  FOR_EACH_IMPL (impl, 0)
    check_result (impl, s1, s2, exp_result);
}

static void
check2 (void)
{
  const char s1_stack[] = ", enable_static, \0, enable_shared, ";
  const size_t s1_byte_count = 18;
  const char *s2_stack = &(s1_stack[s1_byte_count]);
  const size_t s2_byte_count = 18;
  char *exp_result;
  const size_t page_size_real = getpagesize ();

  /* Haystack at end of page.  The following page is protected.  */
  char *s1_page_end = (void *) buf1 + page_size - s1_byte_count;
  strcpy (s1_page_end, s1_stack);

  /* Haystack which crosses a page boundary.
     Note: page_size is at least 2 * getpagesize.  See test_init.  */
  char *s1_page_cross = (void *) buf1 + page_size_real - 8;
  strcpy (s1_page_cross, s1_stack);

  /* Needle at end of page.  The following page is protected.  */
  char *s2_page_end = (void *) buf2 + page_size - s2_byte_count;
  strcpy (s2_page_end, s2_stack);

  /* Needle which crosses a page boundary.
     Note: page_size is at least 2 * getpagesize.  See test_init.  */
  char *s2_page_cross = (void *) buf2 + page_size_real - 8;
  strcpy (s2_page_cross, s2_stack);

  exp_result = stupid_strstr (s1_stack, s2_stack);
  FOR_EACH_IMPL (impl, 0)
    {
      check_result (impl, s1_stack, s2_stack, exp_result);
      check_result (impl, s1_stack, s2_page_end, exp_result);
      check_result (impl, s1_stack, s2_page_cross, exp_result);

      check_result (impl, s1_page_end, s2_stack, exp_result);
      check_result (impl, s1_page_end, s2_page_end, exp_result);
      check_result (impl, s1_page_end, s2_page_cross, exp_result);

      check_result (impl, s1_page_cross, s2_stack, exp_result);
      check_result (impl, s1_page_cross, s2_page_end, exp_result);
      check_result (impl, s1_page_cross, s2_page_cross, exp_result);
    }
}

#define N 1024

static void
pr23637 (void)
{
  char *h = (char*) buf1;
  char *n = (char*) buf2;

  for (int i = 0; i < N; i++)
    {
      n[i] = 'x';
      h[i] = ' ';
      h[i + N] = 'x';
    }

  n[N] = '\0';
  h[N * 2] = '\0';

  /* Ensure we don't match at the first 'x'.  */
  h[0] = 'x';

  char *exp_result = stupid_strstr (h, n);
  FOR_EACH_IMPL (impl, 0)
    check_result (impl, h, n, exp_result);
}

static int
test_main (void)
{
  test_init ();

  check1 ();
  check2 ();
  pr23637 ();

  printf ("%23s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (size_t klen = 2; klen < 32; ++klen)
    for (size_t hlen = 2 * klen; hlen < 16 * klen; hlen += klen)
      {
	do_test (0, 0, hlen, klen, 0);
	do_test (0, 0, hlen, klen, 1);
	do_test (0, 3, hlen, klen, 0);
	do_test (0, 3, hlen, klen, 1);
	do_test (0, 9, hlen, klen, 0);
	do_test (0, 9, hlen, klen, 1);
	do_test (0, 15, hlen, klen, 0);
	do_test (0, 15, hlen, klen, 1);

	do_test (3, 0, hlen, klen, 0);
	do_test (3, 0, hlen, klen, 1);
	do_test (3, 3, hlen, klen, 0);
	do_test (3, 3, hlen, klen, 1);
	do_test (3, 9, hlen, klen, 0);
	do_test (3, 9, hlen, klen, 1);
	do_test (3, 15, hlen, klen, 0);
	do_test (3, 15, hlen, klen, 1);

	do_test (9, 0, hlen, klen, 0);
	do_test (9, 0, hlen, klen, 1);
	do_test (9, 3, hlen, klen, 0);
	do_test (9, 3, hlen, klen, 1);
	do_test (9, 9, hlen, klen, 0);
	do_test (9, 9, hlen, klen, 1);
	do_test (9, 15, hlen, klen, 0);
	do_test (9, 15, hlen, klen, 1);

	do_test (15, 0, hlen, klen, 0);
	do_test (15, 0, hlen, klen, 1);
	do_test (15, 3, hlen, klen, 0);
	do_test (15, 3, hlen, klen, 1);
	do_test (15, 9, hlen, klen, 0);
	do_test (15, 9, hlen, klen, 1);
	do_test (15, 15, hlen, klen, 0);
	do_test (15, 15, hlen, klen, 1);

	do_test (15, 15, hlen + klen * 4, klen * 4, 0);
	do_test (15, 15, hlen + klen * 4, klen * 4, 1);
      }

  do_test (0, 0, page_size - 1, 16, 0);
  do_test (0, 0, page_size - 1, 16, 1);

  return ret;
}

#include <support/test-driver.c>
