/* Test sigaction regression for BZ #23069.
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

#include <signal.h>
#include <unistd.h>

#include <support/check.h>

static void
my_sig_handler (int signum)
{
}

static int
do_test (void)
{
  /* Define a simple signal handler */
  struct sigaction act;
  act.sa_handler = my_sig_handler;
  act.sa_flags = 0;
  sigemptyset (&act.sa_mask);

  /* Set it as SIGUSR1 signal handler */
  TEST_VERIFY_EXIT (sigaction (SIGUSR1, &act, NULL) == 0);

  /* Get SIGUSR1 signal handler */
  TEST_VERIFY_EXIT (sigaction (SIGUSR1, NULL, &act) == 0);

  /* Check it is consistent with the defined one */
  TEST_VERIFY (act.sa_handler == my_sig_handler);
  TEST_VERIFY (!(act.sa_flags & SA_RESETHAND));

  for (int i = 1; i < _NSIG; i++)
    {
      TEST_VERIFY (!sigismember (&act.sa_mask, i));
    }

  return 0;
}

#include <support/test-driver.c>
