/* Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

/* Test case for BZ #15493 */

#include <stdlib.h>
#include <signal.h>
#include <setjmp.h>

static int
do_test (void)
{
  sigjmp_buf sj;
  sigset_t m;

  sigemptyset (&m);
  sigprocmask (SIG_SETMASK, &m, NULL);
  if (sigsetjmp (sj, 0) == 0)
    {
      sigaddset (&m, SIGUSR1);
      sigprocmask (SIG_SETMASK, &m, NULL);
      siglongjmp (sj, 1);
      return EXIT_FAILURE;
    }
  sigprocmask (SIG_SETMASK, NULL, &m);
  return sigismember (&m, SIGUSR1) ? EXIT_SUCCESS : EXIT_FAILURE;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
