/* Helper for exit/dlclose race test (Bug 22180).
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

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <semaphore.h>
#include <unistd.h>
#include <support/check.h>
#include <support/xthread.h>

/* Semaphore defined in executable to ensure we have a happens-before
   between the first function starting and exit being called.  */
extern sem_t order1;

/* Semaphore defined in executable to ensure we have a happens-before
   between the second function starting and the first function returning.  */
extern sem_t order2;

/* glibc function for registering DSO-specific exit functions.  */
extern int __cxa_atexit (void (*func) (void *), void *arg, void *dso_handle);

/* Hidden compiler handle to this shared object.  */
extern void *__dso_handle __attribute__ ((__weak__));

static void
first (void *start)
{
  /* Let the exiting thread run.  */
  sem_post (&order1);

  /* Wait for exiting thread to finish.  */
  sem_wait (&order2);

  printf ("first\n");
}

static void
second (void *start)
{
  /* We may be called from different threads.
     This lock protects called.  */
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  static bool called = false;

  xpthread_mutex_lock (&mtx);
  if (called)
    FAIL_EXIT1 ("second called twice!");

  called = true;
  xpthread_mutex_unlock (&mtx);

  printf ("second\n");
}


__attribute__ ((constructor)) static void
constructor (void)
{
  sem_init (&order1, 0, 0);
  sem_init (&order2, 0, 0);
  __cxa_atexit (second, NULL, __dso_handle);
  __cxa_atexit (first, NULL, __dso_handle);
}
