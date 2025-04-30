/* Tests for pthread_attr_setsigmask_np, pthread_attr_getsigmask_np.
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

/* This thread uses different masked status for SIGUSR1, SIGUSR2,
   SIGHUP to determine if signal masks are applied to new threads as
   expected.  */

#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <support/check.h>
#include <support/xsignal.h>
#include <support/xthread.h>
#include <threads.h>

typedef bool signals[_NSIG];

static const char *
masked_or_unmasked (bool masked)
{
  if (masked)
    return "masked";
  else
    return "unmasked";
}

/* Report an error if ACTUAL_MASK does not match EXPECTED_MASK.
   CONTEXT is used in error messages.  */
static void
check_sigmask (const char *context, signals expected_mask,
               const sigset_t *actual_mask)
{
  for (int sig = 1; sig < _NSIG; ++sig)
    if (sigismember (actual_mask, sig) != expected_mask[sig])
      {
        support_record_failure ();
        printf ("error: %s: signal %d should be %s, but is %s\n",
                context, sig,
                masked_or_unmasked (sigismember (actual_mask, sig)),
                masked_or_unmasked (expected_mask[sig]));
      }
}

/* Report an error if the current thread signal mask does not match
   EXPECTED_MASK.  CONTEXT is used in error messages.  */
static void
check_current_sigmask (const char *context, signals expected_mask)
{
  sigset_t actual_mask;
  xpthread_sigmask (SIG_SETMASK, NULL, &actual_mask);
  check_sigmask (context, expected_mask, &actual_mask);
}

/* Thread start routine which checks the current thread signal mask
   against CLOSURE.  */
static void *
check_sigmask_thread_function (void *closure)
{
  check_current_sigmask ("on thread", closure);
  return NULL;
}

/* Same for C11 threads.  */
static int
check_sigmask_thread_function_c11 (void *closure)
{
  check_current_sigmask ("on C11 thread", closure);
  return 0;
}

/* Launch a POSIX thread with ATTR (which can be NULL) and check that
   it has the expected signal mask.  */
static void
check_posix_thread (pthread_attr_t *attr, signals expected_mask)
{
  xpthread_join (xpthread_create (attr, check_sigmask_thread_function,
                                  expected_mask));
}

/* Launch a C11 thread and check that it has the expected signal
   mask.  */
static void
check_c11_thread (signals expected_mask)
{
  thrd_t thr;
  TEST_VERIFY_EXIT (thrd_create (&thr, check_sigmask_thread_function_c11,
                                 expected_mask) == thrd_success);
  TEST_VERIFY_EXIT (thrd_join (thr, NULL) == thrd_success);
}

static int
do_test (void)
{
  check_current_sigmask ("initial mask", (signals) { false, });
  check_posix_thread (NULL, (signals) { false, });
  check_c11_thread ((signals) { false, });

  sigset_t set;
  sigemptyset (&set);
  sigaddset (&set, SIGUSR1);
  xpthread_sigmask (SIG_SETMASK, &set, NULL);
  check_current_sigmask ("SIGUSR1 masked", (signals) { [SIGUSR1] = true, });
  /* The signal mask is inherited by the new thread.  */
  check_posix_thread (NULL, (signals) { [SIGUSR1] = true, });
  check_c11_thread ((signals) { [SIGUSR1] = true, });

  pthread_attr_t attr;
  xpthread_attr_init (&attr);
  TEST_COMPARE (pthread_attr_getsigmask_np (&attr, &set),
                PTHREAD_ATTR_NO_SIGMASK_NP);
  /* By default, the signal mask is inherited (even with an explicit
     thread attribute).  */
  check_posix_thread (&attr, (signals) { [SIGUSR1] = true, });

  /* Check that pthread_attr_getsigmask_np can obtain the signal
     mask.  */
  sigemptyset (&set);
  sigaddset (&set, SIGUSR2);
  TEST_COMPARE (pthread_attr_setsigmask_np (&attr, &set), 0);
  sigemptyset (&set);
  TEST_COMPARE (pthread_attr_getsigmask_np (&attr, &set), 0);
  check_sigmask ("pthread_attr_getsigmask_np", (signals) { [SIGUSR2] = true, },
                 &set);

  /* Check that a thread is launched with the configured signal
     mask.  */
  check_current_sigmask ("SIGUSR1 masked", (signals) { [SIGUSR1] = true, });
  check_posix_thread (&attr, (signals) { [SIGUSR2] = true, });
  check_current_sigmask ("SIGUSR1 masked", (signals) { [SIGUSR1] = true, });

  /* But C11 threads remain at inheritance.  */
  check_c11_thread ((signals) { [SIGUSR1] = true, });

  /* Check that filling the original signal set does not affect thread
     creation.  */
  sigfillset (&set);
  check_posix_thread (&attr, (signals) { [SIGUSR2] = true, });

  /* Check that clearing the signal in the attribute restores
     inheritance.  */
  TEST_COMPARE (pthread_attr_setsigmask_np (&attr, NULL), 0);
  TEST_COMPARE (pthread_attr_getsigmask_np (&attr, &set),
                PTHREAD_ATTR_NO_SIGMASK_NP);
  check_posix_thread (&attr, (signals) { [SIGUSR1] = true, });

  /* Mask SIGHUP via the default thread attribute.  */
  sigemptyset (&set);
  sigaddset (&set, SIGHUP);
  TEST_COMPARE (pthread_attr_setsigmask_np (&attr, &set), 0);
  TEST_COMPARE (pthread_setattr_default_np (&attr), 0);

  /* Check that the mask was applied to the default attribute.  */
  xpthread_attr_destroy (&attr);
  TEST_COMPARE (pthread_getattr_default_np (&attr), 0);
  sigaddset (&set, SIGHUP);
  TEST_COMPARE (pthread_attr_getsigmask_np (&attr, &set), 0);
  check_sigmask ("default attribute", (signals) { [SIGHUP] = true, }, &set);
  xpthread_attr_destroy (&attr);

  /* Check that the default attribute is applied.  */
  check_posix_thread (NULL, (signals) { [SIGHUP] = true, });
  check_c11_thread ((signals) { [SIGHUP] = true, });

  /* An explicit attribute with no signal mask triggers inheritance
     even if the default has been changed.  */
  xpthread_attr_init (&attr);
  check_posix_thread (&attr, (signals) { [SIGUSR1] = true, });

  /* Explicitly setting the signal mask affects the new thread even
     with a default attribute.  */
  sigemptyset (&set);
  sigaddset (&set, SIGUSR2);
  TEST_COMPARE (pthread_attr_setsigmask_np (&attr, &set), 0);
  check_posix_thread (&attr, (signals) { [SIGUSR2] = true, });

  /* Resetting the default attribute brings back the old inheritance
     behavior.  */
  xpthread_attr_destroy (&attr);
  xpthread_attr_init (&attr);
  TEST_COMPARE (pthread_setattr_default_np (&attr), 0);
  xpthread_attr_destroy (&attr);
  check_posix_thread (NULL, (signals) { [SIGUSR1] = true, });
  check_c11_thread ((signals) { [SIGUSR1] = true, });

  return 0;
}

#include <support/test-driver.c>
