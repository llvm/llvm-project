/* Measure __strcpy_chk functions.
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

#ifndef STRCPY_RESULT
# define STRCPY_RESULT(dst, len) dst
# define TEST_MAIN
# define TEST_NAME "strcpy_chk"
# include "bench-string.h"

/* This test case implicitly tests the availability of the __chk_fail
   symbol, which is part of the public ABI and may be used
   externally. */
extern void __attribute__ ((noreturn)) __chk_fail (void);
char *simple_strcpy_chk (char *, const char *, size_t);
extern char *normal_strcpy (char *, const char *, size_t)
  __asm ("strcpy");
extern char *__strcpy_chk (char *, const char *, size_t);

IMPL (simple_strcpy_chk, 0)
IMPL (normal_strcpy, 1)
IMPL (__strcpy_chk, 2)

char *
simple_strcpy_chk (char *dst, const char *src, size_t len)
{
  char *ret = dst;
  if (! len)
    __chk_fail ();
  while ((*dst++ = *src++) != '\0')
    if (--len == 0)
      __chk_fail ();
  return ret;
}
#endif

#include <fcntl.h>
#include <paths.h>
#include <setjmp.h>
#include <signal.h>

#include <support/support.h>

volatile int chk_fail_ok;
jmp_buf chk_fail_buf;

static void
handler (int sig)
{
  if (chk_fail_ok)
    {
      chk_fail_ok = 0;
      longjmp (chk_fail_buf, 1);
    }
  else
    _exit (127);
}

typedef char *(*proto_t) (char *, const char *, size_t);

static void
do_one_test (impl_t *impl, char *dst, const char *src,
	     size_t len, size_t dlen)
{
  char *res;
  size_t i, iters = INNER_LOOP_ITERS8;
  timing_t start, stop, cur;

  if (dlen <= len)
    {
      if (impl->test == 1)
	return;

      chk_fail_ok = 1;
      if (setjmp (chk_fail_buf) == 0)
	{
	  res = CALL (impl, dst, src, dlen);
	  printf ("*** Function %s (%zd; %zd) did not __chk_fail\n",
		  impl->name, len, dlen);
	  chk_fail_ok = 0;
	  ret = 1;
	}
      return;
    }
  else
    res = CALL (impl, dst, src, dlen);

  if (res != STRCPY_RESULT (dst, len))
    {
      printf ("Wrong result in function %s %p %p\n", impl->name,
	      res, STRCPY_RESULT (dst, len));
      ret = 1;
      return;
    }

  if (strcmp (dst, src) != 0)
    {
      printf ("Wrong result in function %s dst \"%s\" src \"%s\"\n",
	      impl->name, dst, src);
      ret = 1;
      return;
    }

  TIMING_NOW (start);
  for (i = 0; i < iters; ++i)
    {
      CALL (impl, dst, src, dlen);
    }
  TIMING_NOW (stop);

  TIMING_DIFF (cur, start, stop);

  TIMING_PRINT_MEAN ((double) cur, (double) iters);
}

static void
do_test (size_t align1, size_t align2, size_t len, size_t dlen, int max_char)
{
  size_t i;
  char *s1, *s2;

  align1 &= 7;
  if (align1 + len >= page_size)
    return;

  align2 &= 7;
  if (align2 + len >= page_size)
    return;

  s1 = (char *) buf1 + align1;
  s2 = (char *) buf2 + align2;

  for (i = 0; i < len; i++)
    s1[i] = 32 + 23 * i % (max_char - 32);
  s1[len] = 0;

  if (dlen > len)
    printf ("Length %4zd, alignment %2zd/%2zd:", len, align1, align2);

  FOR_EACH_IMPL (impl, 0)
    do_one_test (impl, s2, s1, len, dlen);

  if (dlen > len)
    putchar ('\n');
}

static int
test_main (void)
{
  size_t i;

  set_fortify_handler (handler);

  test_init ();

  printf ("%23s", "");
  FOR_EACH_IMPL (impl, 0)
    printf ("\t%s", impl->name);
  putchar ('\n');

  for (i = 0; i < 16; ++i)
    {
      do_test (0, 0, i, i + 1, 127);
      do_test (0, 0, i, i + 1, 255);
      do_test (0, i, i, i + 1, 127);
      do_test (i, 0, i, i + 1, 255);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, 8 << i, (8 << i) + 1, 127);
      do_test (8 - i, 2 * i, (8 << i), (8 << i) + 1, 127);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 2 * i, (8 << i), (8 << i) + 1, 127);
      do_test (2 * i, i, (8 << i), (8 << i) + 1, 255);
      do_test (i, i, (8 << i), (8 << i) + 1, 127);
      do_test (i, i, (8 << i), (8 << i) + 1, 255);
    }

  for (i = 0; i < 16; ++i)
    {
      do_test (0, 0, i, i + 256, 127);
      do_test (0, 0, i, i + 256, 255);
      do_test (0, i, i, i + 256, 127);
      do_test (i, 0, i, i + 256, 255);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, 8 << i, (8 << i) + 256, 127);
      do_test (8 - i, 2 * i, (8 << i), (8 << i) + 256, 127);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 2 * i, (8 << i), (8 << i) + 256, 127);
      do_test (2 * i, i, (8 << i), (8 << i) + 256, 255);
      do_test (i, i, (8 << i), (8 << i) + 256, 127);
      do_test (i, i, (8 << i), (8 << i) + 256, 255);
    }

  for (i = 0; i < 16; ++i)
    {
      do_test (0, 0, i, i, 127);
      do_test (0, 0, i, i + 2, 255);
      do_test (0, i, i, i + 3, 127);
      do_test (i, 0, i, i + 4, 255);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (0, 0, 8 << i, (8 << i) - 15, 127);
      do_test (8 - i, 2 * i, (8 << i), (8 << i) + 5, 127);
    }

  for (i = 1; i < 8; ++i)
    {
      do_test (i, 2 * i, (8 << i), (8 << i) + i, 127);
      do_test (2 * i, i, (8 << i), (8 << i) + (i - 1), 255);
      do_test (i, i, (8 << i), (8 << i) + i + 2, 127);
      do_test (i, i, (8 << i), (8 << i) + i + 3, 255);
    }

  return 0;
}

#include <support/test-driver.c>
