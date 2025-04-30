/* Test reporting of out-of-bounds access for dynamic arrays.
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

#include "tst-dynarray-shared.h"

#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <support/capture_subprocess.h>
#include <support/check.h>

/* Run CALLBACK and check that the data on standard error equals
   EXPECTED.  */
static void
check (const char *test, void (*callback) (void *), size_t index,
       const char *expected)
{
  struct support_capture_subprocess result
    = support_capture_subprocess (callback, &index);
  if (strcmp (result.err.buffer, expected) != 0)
    {
      support_record_failure ();
      printf ("error: test %s (%zu) unexpected standard error data\n"
              "  expected: %s\n"
              "  actual:   %s\n",
              test, index, expected, result.err.buffer);
    }
  TEST_VERIFY (strlen (result.out.buffer) == 0);
  TEST_VERIFY (WIFSIGNALED (result.status));
  if (WIFSIGNALED (result.status))
    TEST_VERIFY (WTERMSIG (result.status) == SIGABRT);
  support_capture_subprocess_free (&result);
}

/* Try indexing an empty array.  */
static void
test_empty (void *closure)
{
  size_t *pindex = closure;
  struct dynarray_int dyn;
  dynarray_int_init (&dyn);
  dynarray_int_at (&dyn, *pindex);
}

/* Try indexing a one-element array.  */
static void
test_one (void *closure)
{
  size_t *pindex = closure;
  struct dynarray_int dyn;
  dynarray_int_init (&dyn);
  TEST_VERIFY (dynarray_int_resize (&dyn, 1));
  dynarray_int_at (&dyn, *pindex);
}

/* Try indexing a longer array.  */
static void
test_many (void *closure)
{
  size_t *pindex = closure;
  struct dynarray_int dyn;
  dynarray_int_init (&dyn);
  TEST_VERIFY (dynarray_int_resize (&dyn, 5371));
  dynarray_int_at (&dyn, *pindex);
}

/* (size_t) -1 for use in string literals.  */
#if SIZE_WIDTH == 32
# define MINUS_1 "4294967295"
#elif SIZE_WIDTH == 64
# define MINUS_1 "18446744073709551615"
#else
# error "unknown value for SIZE_WIDTH"
#endif

static int
do_test (void)
{
  TEST_VERIFY (setenv ("LIBC_FATAL_STDERR_", "1", 1) == 0);

  check ("test_empty", test_empty, 0,
         "Fatal glibc error: array index 0 not less than array length 0\n");
  check ("test_empty", test_empty, 1,
         "Fatal glibc error: array index 1 not less than array length 0\n");
  check ("test_empty", test_empty, -1,
         "Fatal glibc error: array index " MINUS_1
         " not less than array length 0\n");

  check ("test_one", test_one, 1,
         "Fatal glibc error: array index 1 not less than array length 1\n");
  check ("test_one", test_one, 2,
         "Fatal glibc error: array index 2 not less than array length 1\n");
  check ("test_one", test_one, -1,
         "Fatal glibc error: array index " MINUS_1
         " not less than array length 1\n");

  check ("test_many", test_many, 5371,
         "Fatal glibc error: array index 5371"
         " not less than array length 5371\n");
  check ("test_many", test_many, 5372,
         "Fatal glibc error: array index 5372"
         " not less than array length 5371\n");
  check ("test_many", test_many, -1,
         "Fatal glibc error: array index " MINUS_1
         " not less than array length 5371\n");

  return 0;
}

#include <support/test-driver.c>
