/* Test CET property note parser for [BZ #23467].
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <support/check.h>

extern void bar (void);

void
__attribute__ ((noclone, noinline))
test (void (*func_p) (void))
{
  func_p ();
}

/* bar contains an IBT violation if it is called indirectly via a
   function pointer.  On IBT machines, it should lead to segfault
   unless IBT is disabled by error.  */

static void
sig_handler (int signo)
{
  exit (EXIT_SUCCESS);
}

static int
do_test (void)
{
  char buf[4];

  if (scanf ("%3s", buf) != 1)
    FAIL_UNSUPPORTED ("IBT not supported");

  if (strcmp (buf, "IBT") != 0)
    FAIL_UNSUPPORTED ("IBT not supported");

  TEST_VERIFY_EXIT (signal (SIGSEGV, &sig_handler) != SIG_ERR);

  /* Call bar via a function pointer to force an IBT violation.  */
  test (bar);

  return EXIT_FAILURE;
}

#include <support/test-driver.c>
