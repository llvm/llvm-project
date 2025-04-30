/* Test for exit/dlclose race (Bug 22180).
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

/* This file must be run from within a directory called "stdlib".  */

/* This test verifies that when dlopen in one thread races against exit
   in another thread, we don't call registered destructor twice.

   Expected result:
     second
     first
     ... clean termination
*/

#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <support/check.h>
#include <support/xdlfcn.h>
#include <support/xthread.h>

/* Semaphore to ensure we have a happens-before between the first function
   starting and exit being called.  */
sem_t order1;

/* Semaphore to ensure we have a happens-before between the second function
   starting and the first function returning.  */
sem_t order2;

void *
exit_thread (void *arg)
{
  /* Wait for the dlclose to start...  */
  sem_wait (&order1);
  /* Then try to run the exit sequence which should call all
     __cxa_atexit registered functions and in parallel with
     the executing dlclose().  */
  exit (0);
}


void
last (void)
{
  /* Let dlclose thread proceed.  */
  sem_post (&order2);
}

int
main (void)
{
  void *dso;
  pthread_t thread;

  atexit (last);

  dso = xdlopen ("$ORIGIN/test-dlclose-exit-race-helper.so",
		 RTLD_NOW|RTLD_GLOBAL);
  thread = xpthread_create (NULL, exit_thread, NULL);

  xdlclose (dso);
  xpthread_join (thread);

  FAIL_EXIT1 ("Did not terminate via exit(0) in exit_thread() as expected.");
}
