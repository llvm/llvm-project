/* Test for the long double variants of obstrack*printf functions.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <malloc.h>
#include <obstack.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include <support/check.h>

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free

static void
do_test_call (void *last, ...)
{
  const char *expected = "-1.000000000000000000000000000000";
  char *actual = NULL;
  long double ld = -1;
  struct obstack ob;
  va_list ap;

  obstack_init (&ob);
  OBSTACK_FUNCTION OBSTACK_FUNCTION_PARAMS;
  actual = (char *) obstack_finish (&ob);
  TEST_VERIFY (strncmp (expected, actual, 33) == 0);
  obstack_free (&ob, NULL);
  actual = NULL;

  obstack_init (&ob);
  va_start (ap, last);
  VOBSTACK_FUNCTION VOBSTACK_FUNCTION_PARAMS;
  va_end (ap);
  actual = (char *) obstack_finish (&ob);
  TEST_VERIFY (strncmp (expected, actual, 33) == 0);
  obstack_free (&ob, NULL);
  actual = NULL;
}

static int
do_test (void)
{
  long double ld = -1;
  do_test_call (NULL, ld);
  return 0;
}

#include <support/test-driver.c>
