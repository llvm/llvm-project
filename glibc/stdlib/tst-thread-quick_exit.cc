/* Bug 20198: Do not call object destructors at exit.
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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct A
{
  ~A () { abort (); }
};

thread_local A a1;
thread_local A a2;

void
__attribute__ ((noinline, noclone))
optimization_barrier (A &)
{
}

/* Call std::quick_exit from a non-main thread.  */
void *
non_main_thread (void *)
{
  optimization_barrier (a1);
  /* The C++11 standard in 18.5.12 says:
     "Objects shall not be destroyed as a result of calling
      quick_exit."
     If quick_exit calls the destructors the test aborts.  */
  quick_exit (0);
}

static int
do_test()
{
  optimization_barrier (a2);
  pthread_t thr;
  int ret = pthread_create (&thr, NULL, non_main_thread, NULL);
  if (ret != 0)
    {
      errno = ret;
      printf ("error: pthread_create: %m\n");
    }
  pthread_join (thr, NULL);
  return 1;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
