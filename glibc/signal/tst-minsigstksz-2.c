/* Tests of signal delivery on an alternate stack (abort).
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

#include <support/xsignal.h>
#include <support/support.h>
#include <support/check.h>
#include <stdlib.h>

/* C2011 7.4.1.1p5 specifies that only the following operations are
   guaranteed to be well-defined inside an asynchronous signal handler:
     * any operation on a lock-free atomic object
     * assigning a value to an object declared as volatile sig_atomic_t
     * calling abort, _Exit, quick_exit, or signal
       * signal may only be called with its first argument equal to the
         number of the signal that caused the handler to be called

   We use this list as a guideline for the set of operations that ought
   also to be safe in a _synchronous_ signal delivered on an alternate
   signal stack with only MINSIGSTKSZ bytes of space.

   This test program tests calls to abort.  Note that it does _not_
   install a handler for SIGABRT, because that signal would also be
   delivered on the alternate stack and MINSIGSTKSZ does not provide
   enough space for delivery of nested signals.  */

static void
handler (int unused)
{
  abort ();
}

int
do_test (void)
{
  void *sstk = xalloc_sigstack (0);
  struct sigaction sa;

  sa.sa_handler = handler;
  sa.sa_flags   = SA_RESTART | SA_ONSTACK;
  sigfillset (&sa.sa_mask);
  if (sigaction (SIGUSR1, &sa, 0))
    FAIL_RET ("sigaction (SIGUSR1, handler): %m\n");

  raise (SIGUSR1);

  xfree_sigstack (sstk);
  FAIL_RET ("test process was not terminated by abort in signal handler");
}

#define EXPECTED_SIGNAL SIGABRT
#include <support/test-driver.c>
