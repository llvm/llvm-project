/* Tests of signal delivery on an alternate stack (nonlethal).
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

   This test program tests all of the above operations that do not,
   one way or another, cause the program to be terminated.  */

/* We do not try to test atomic operations exhaustively, only a simple
   atomic counter increment.  This is only safe if atomic_[u]int is
   unconditionally lock-free.  */
#ifdef __STDC_NO_ATOMICS__
# define TEST_ATOMIC_OPS 0
#else
# include <stdatomic.h>
# if ATOMIC_INT_LOCK_FREE != 2
#  define TEST_ATOMIC_OPS 0
# else
#  define TEST_ATOMIC_OPS 1
# endif
#endif

static volatile sig_atomic_t signal_flag = 0;
static volatile sig_atomic_t signal_err = 0;
static void
handler_set_flag (int unused)
{
  signal_flag = 1;
}

static void
handler_set_flag_once (int sig)
{
  signal_flag = 1;
  if (signal (sig, SIG_IGN) == SIG_ERR)
    /* It is not safe to call FAIL_EXIT1 here.  Set another flag instead.  */
    signal_err = 1;
}

#if TEST_ATOMIC_OPS
static atomic_uint signal_count = 0;
static void
handler_count_up_1 (int unused)
{
  atomic_fetch_add (&signal_count, 1);
}
#endif

int
do_test (void)
{
  void *sstk = xalloc_sigstack (0);
  struct sigaction sa;

  /* Test 1: setting a volatile sig_atomic_t flag.  */
  sa.sa_handler = handler_set_flag;
  sa.sa_flags   = SA_RESTART | SA_ONSTACK;
  sigfillset (&sa.sa_mask);
  if (sigaction (SIGUSR1, &sa, 0))
    FAIL_EXIT1 ("sigaction (SIGUSR1, handler_set_flag): %m\n");

  TEST_VERIFY_EXIT (signal_flag == 0);
  raise (SIGUSR1);
  TEST_VERIFY_EXIT (signal_flag == 1);
  signal_flag = 0;
  raise (SIGUSR1);
  TEST_VERIFY_EXIT (signal_flag == 1);
  signal_flag = 0;

  /* Test 1: setting a volatile sig_atomic_t flag and then ignoring
     further delivery of the signal. */
  sa.sa_handler = handler_set_flag_once;
  if (sigaction (SIGUSR1, &sa, 0))
    FAIL_EXIT1 ("sigaction (SIGUSR1, handler_set_flag_once): %m\n");

  raise (SIGUSR1);
  TEST_VERIFY_EXIT (signal_flag == 1);
  /* Note: if signal_err is 1, a system call failed, but we can't
     report the error code because errno is indeterminate.  */
  TEST_VERIFY_EXIT (signal_err == 0);

  signal_flag = 0;
  raise (SIGUSR1);
  TEST_VERIFY_EXIT (signal_flag == 0);
  TEST_VERIFY_EXIT (signal_err == 0);

#if TEST_ATOMIC_OPS
  sa.sa_handler = handler_count_up_1;
  if (sigaction (SIGUSR1, &sa, 0))
    FAIL_EXIT1 ("sigaction (SIGUSR1, handler_count_up_1): %m\n");

  raise (SIGUSR1);
  TEST_VERIFY_EXIT (atomic_load (&signal_count) == 1);
  raise (SIGUSR1);
  TEST_VERIFY_EXIT (atomic_load (&signal_count) == 2);
#endif

  xfree_sigstack (sstk);
  return 0;
}

#include <support/test-driver.c>
