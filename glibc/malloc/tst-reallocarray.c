/* Test for reallocarray.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <malloc.h>
#include <string.h>
#include <support/check.h>
#include <libc-diag.h>

static void *
reallocarray_nowarn (void *ptr, size_t nmemb, size_t size)
{
#if __GNUC_PREREQ (7, 0)
  /* GCC 7 warns about too-large allocations; here we want to test
     that they fail.  */
  DIAG_IGNORE_NEEDS_COMMENT (7, "-Walloc-size-larger-than=");
#endif
  void *ret = reallocarray (ptr, nmemb, size);
#if __GNUC_PREREQ (7, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif
  return ret;
}


static int
do_test (void)
{
  void *ptr = NULL;
  void *ptr2 = NULL;
  unsigned char *c;
  size_t i;
  int ok;
  const size_t max = ~(size_t)0;
  size_t a, b;

  /* Test overflow detection.  */
  errno = 0;
  ptr = reallocarray_nowarn (NULL, max, 2);
  TEST_VERIFY (!ptr);
  TEST_VERIFY (errno == ENOMEM);

  errno = 0;
  ptr = reallocarray_nowarn (NULL, 2, max);
  TEST_VERIFY (!ptr);
  TEST_VERIFY (errno == ENOMEM);

  a = 65537;
  b = max/65537 + 1;
  errno = 0;
  ptr = reallocarray_nowarn (NULL, a, b);
  TEST_VERIFY (!ptr);
  TEST_VERIFY (errno == ENOMEM);

  errno = 0;
  ptr = reallocarray_nowarn (NULL, b, a);
  TEST_VERIFY (!ptr);
  TEST_VERIFY (errno == ENOMEM);

  /* Test realloc-like behavior.  */
  /* Allocate memory like malloc.  */
  ptr = reallocarray (NULL, 10, 2);
  TEST_VERIFY_EXIT (ptr);
  TEST_VERIFY_EXIT (malloc_usable_size (ptr) >= 10*2);

  memset (ptr, 0xAF, 10*2);

  /* Enlarge buffer.   */
  ptr2 = reallocarray (ptr, 20, 2);
  TEST_VERIFY (ptr2);
  if (ptr2)
    ptr = ptr2;
  TEST_VERIFY (malloc_usable_size (ptr) >= 20*2);

  c = ptr;
  ok = 1;
  for (i = 0; i < 10*2; ++i)
    {
      if (c[i] != 0xAF)
        ok = 0;
    }
  TEST_VERIFY (ok);

  /* Decrease buffer size.  */
  ptr2 = reallocarray (ptr, 5, 3);
  TEST_VERIFY (ptr2);
  if (ptr2)
    ptr = ptr2;
  TEST_VERIFY_EXIT (malloc_usable_size (ptr) >= 5*3);

  c = ptr;
  ok = 1;
  for (i = 0; i < 5*3; ++i)
    {
      if (c[i] != 0xAF)
        ok = 0;
    }
  TEST_VERIFY (ok);

  /* Overflow should leave buffer untouched.  */
  errno = 0;
  ptr2 = reallocarray_nowarn (ptr, 2, ~(size_t)0);
  TEST_VERIFY (!ptr2);
  TEST_VERIFY (errno == ENOMEM);

  c = ptr;
  ok = 1;
  for (i = 0; i < 5*3; ++i)
    {
      if (c[i] != 0xAF)
        ok = 0;
    }
  TEST_VERIFY (ok);

  free (ptr);

  return 0;
}

#include <support/test-driver.c>
