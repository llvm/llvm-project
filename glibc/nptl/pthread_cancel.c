/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <signal.h>
#include <stdlib.h>
#include "pthreadP.h"
#include <atomic.h>
#include <sysdep.h>
#include <unistd.h>
#include <unwind-link.h>
#include <stdio.h>
#include <gnu/lib-names.h>
#include <sys/single_threaded.h>

/* For asynchronous cancellation we use a signal.  */
static void
sigcancel_handler (int sig, siginfo_t *si, void *ctx)
{
  /* Safety check.  It would be possible to call this function for
     other signals and send a signal from another process.  This is not
     correct and might even be a security problem.  Try to catch as
     many incorrect invocations as possible.  */
  if (sig != SIGCANCEL
      || si->si_pid != __getpid()
      || si->si_code != SI_TKILL)
    return;

  struct pthread *self = THREAD_SELF;

  int ch = atomic_load_relaxed (&self->cancelhandling);
  /* Cancelation not enabled, not cancelled, or already exitting.  */
  if (self->cancelstate == PTHREAD_CANCEL_DISABLE
      || (ch & CANCELED_BITMASK) == 0
      || (ch & EXITING_BITMASK) != 0)
    return;

  /* Set the return value.  */
  THREAD_SETMEM (self, result, PTHREAD_CANCELED);
  /* Make sure asynchronous cancellation is still enabled.  */
  if (self->canceltype == PTHREAD_CANCEL_ASYNCHRONOUS)
    __do_cancel ();
}

int
__pthread_cancel (pthread_t th)
{
  volatile struct pthread *pd = (volatile struct pthread *) th;

  /* Make sure the descriptor is valid.  */
  if (INVALID_TD_P (pd))
    /* Not a valid thread handle.  */
    return ESRCH;

  static int init_sigcancel = 0;
  if (atomic_load_relaxed (&init_sigcancel) == 0)
    {
      struct sigaction sa;
      sa.sa_sigaction = sigcancel_handler;
      /* The signal handle should be non-interruptible to avoid the risk of
	 spurious EINTR caused by SIGCANCEL sent to process or if
	 pthread_cancel() is called while cancellation is disabled in the
	 target thread.  */
      sa.sa_flags = SA_SIGINFO | SA_RESTART;
      __sigemptyset (&sa.sa_mask);
      __libc_sigaction (SIGCANCEL, &sa, NULL);
      atomic_store_relaxed (&init_sigcancel, 1);
    }

#ifdef SHARED
  /* Trigger an error if libgcc_s cannot be loaded.  */
  {
    struct unwind_link *unwind_link = __libc_unwind_link_get ();
    if (unwind_link == NULL)
      __libc_fatal (LIBGCC_S_SO
		    " must be installed for pthread_cancel to work\n");
  }
#endif

  int oldch = atomic_fetch_or_acquire (&pd->cancelhandling, CANCELED_BITMASK);
  if ((oldch & CANCELED_BITMASK) != 0)
    return 0;

  if (pd == THREAD_SELF)
    {
      /* A single-threaded process should be able to kill itself, since there
	 is nothing in the POSIX specification that says that it cannot.  So
	 we set multiple_threads to true so that cancellation points get
	 executed.  */
      THREAD_SETMEM (THREAD_SELF, header.multiple_threads, 1);
#ifndef TLS_MULTIPLE_THREADS_IN_TCB
      __libc_multiple_threads = 1;
#endif

      THREAD_SETMEM (pd, result, PTHREAD_CANCELED);
      if (pd->cancelstate == PTHREAD_CANCEL_ENABLE
	  && pd->canceltype == PTHREAD_CANCEL_ASYNCHRONOUS)
	__do_cancel ();
      return 0;
    }

  return __pthread_kill_internal (th, SIGCANCEL);
}
versioned_symbol (libc, __pthread_cancel, pthread_cancel, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, __pthread_cancel, pthread_cancel, GLIBC_2_0);
#endif

/* Ensure that the unwinder is always linked in (the __pthread_unwind
   reference from __do_cancel is weak).  Use ___pthread_unwind_next
   (three underscores) to produce a strong reference to the same
   file.  */
PTHREAD_STATIC_FN_REQUIRE (___pthread_unwind_next)
