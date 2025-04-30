/* Time-triggered process termination.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/xthread.h>
#include <support/xsignal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <time.h>

static void *
delayed_exit_thread (void *seconds_as_ptr)
{
  int seconds = (uintptr_t) seconds_as_ptr;
  struct timespec delay = { seconds, 0 };
  struct timespec remaining = { 0 };
  if (nanosleep (&delay, &remaining) != 0)
    FAIL_EXIT1 ("nanosleep: %m");
  /* Exit the process sucessfully.  */
  exit (0);
  return NULL;
}

void
delayed_exit (int seconds)
{
  /* Create the new thread with all signals blocked.  */
  sigset_t all_blocked;
  sigfillset (&all_blocked);
  sigset_t old_set;
  xpthread_sigmask (SIG_SETMASK, &all_blocked, &old_set);
  /* Create a detached thread. */
  pthread_t thr = xpthread_create
    (NULL, delayed_exit_thread, (void *) (uintptr_t) seconds);
  xpthread_detach (thr);
  /* Restore the original signal mask.  */
  xpthread_sigmask (SIG_SETMASK, &old_set, NULL);
}
