/* Check that module __end_fct is not invoked when the init function fails.
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
#include <iconv.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <sys/auxv.h>

/* Test GCONV_PATH to the directory containing the program
   executable.  */
static void
activate_test_gconv_modules (void)
{
  unsigned long ptr = getauxval (AT_EXECFN);
  if (ptr == 0)
    {
      printf ("warning: AT_EXECFN not support, cannot run test\n");
      exit (EXIT_UNSUPPORTED);
    }
  char *test_program_directory = dirname (xstrdup ((const char *) ptr));
  TEST_VERIFY (setenv ("GCONV_PATH", test_program_directory, 1) == 0);
  free (test_program_directory);
}

static int
do_test (void)
{
  activate_test_gconv_modules ();

  TEST_VERIFY (iconv_open ("UTF-8", "tst-gconv-init-failure//")
               == (iconv_t) -1);
  if (errno != ENOMEM)
    FAIL_EXIT1 ("unexpected iconv_open error: %m");

  return 0;
}

#include <support/test-driver.c>
