/* Tests for sigisemptyset/sigorset/sigandset.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <support/check.h>

static int
do_test (void)
{
  {
    sigset_t set;
    sigemptyset (&set);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  {
    sigset_t set;
    sigfillset (&set);
    TEST_COMPARE (sigisemptyset (&set), 0);
  }

  {
    sigset_t setfill, setempty, set;
    sigfillset (&setfill);
    sigemptyset (&setempty);

    sigorset (&set, &setfill, &setempty);
    TEST_COMPARE (sigisemptyset (&set), 0);

    sigandset (&set, &setfill, &setempty);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  /* Ensure current SIG_BLOCK mask empty.  */
  {
    sigset_t set;
    sigemptyset (&set);
    TEST_COMPARE (sigprocmask (SIG_BLOCK, &set, 0), 0);
  }

  {
    sigset_t set;
    sigemptyset (&set);
    TEST_COMPARE (sigprocmask (SIG_BLOCK, 0, &set), 0);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  {
    sigset_t set;
    sigfillset (&set);
    TEST_COMPARE (sigprocmask (SIG_BLOCK, 0, &set), 0);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  /* Block all signals.  */
  {
    sigset_t set;
    sigfillset (&set);
    TEST_COMPARE (sigprocmask (SIG_BLOCK, &set, 0), 0);
  }

  {
    sigset_t set;
    sigemptyset (&set);
    TEST_COMPARE (sigpending (&set), 0);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  {
    sigset_t set;
    sigfillset (&set);
    TEST_COMPARE (sigpending (&set), 0);
    TEST_COMPARE (sigisemptyset (&set), 1);
  }

  return 0;
}

#include <support/test-driver.c>
