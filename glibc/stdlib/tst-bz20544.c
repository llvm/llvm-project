/* Verify atexit, on_exit, etc. abort on NULL function pointer.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>

extern int __cxa_atexit (void (*func) (void *), void *arg, void *d);
extern int __cxa_at_quick_exit (void (*func) (void *), void *arg, void *d);

/* GCC "knows" that atexit and on_exit should not be called with NULL
   function pointer, and emits diagnostics if we try to do so.
   Presumably it could emit a trap and drop the call altogether.

   The aliases below are intended to bypass this.  */

extern int atexit_alias (void (*) (void)) __asm__ ("atexit");
extern int at_quick_exit_alias (void (*) (void)) __asm__ ("at_quick_exit");
extern int on_exit_alias (void (*) (void), void *) __asm__ ("on_exit");


static void
test_bz20544_atexit (void *closure)
{
  atexit_alias (NULL);  /* Should assert.  */
  exit (EXIT_FAILURE);
}

static void
test_bz20544_at_quick_exit (void *closure)
{
  at_quick_exit_alias (NULL);  /* Should assert.  */
  exit (EXIT_FAILURE);
}

static void
test_bz20544_on_exit (void *closure)
{
  on_exit_alias (NULL, NULL);  /* Should assert.  */
  exit (EXIT_FAILURE);
}

static void
test_bz20544_cxa_atexit (void *closure)
{
  __cxa_atexit (NULL, NULL, NULL);  /* Should assert.  */
  exit (EXIT_FAILURE);
}

static void
test_bz20544_cxa_at_quick_exit (void *closure)
{
  __cxa_at_quick_exit (NULL, NULL, NULL);  /* Should assert.  */
  exit (EXIT_FAILURE);
}

static void
test_one_fn (void (*test_fn) (void *))
{
  const char expected_error[] = "Assertion `func != NULL' failed.\n";
  struct support_capture_subprocess result;
  result = support_capture_subprocess (test_fn, NULL);
  support_capture_subprocess_check (&result, "bz20544", -SIGABRT,
                                    sc_allow_stderr);

  if (strstr (result.err.buffer, expected_error) == NULL)
    {
      support_record_failure ();
      printf ("Did not find expected string in error output:\n"
              "  expected: >>>%s<<<\n"
              "  actual:   >>>%s<<<\n",
              expected_error, result.err.buffer);
    }

  support_capture_subprocess_free (&result);
}

static int
do_test (void)
{
#if defined (NDEBUG)
  FAIL_UNSUPPORTED ("Assertions disabled (NDEBUG). "
                    "Can't verify that assertions fire.");
#endif
  test_one_fn (test_bz20544_atexit);
  test_one_fn (test_bz20544_at_quick_exit);
  test_one_fn (test_bz20544_on_exit);
  test_one_fn (test_bz20544_cxa_atexit);
  test_one_fn (test_bz20544_cxa_at_quick_exit);

  return 0;
}

#include <support/test-driver.c>
