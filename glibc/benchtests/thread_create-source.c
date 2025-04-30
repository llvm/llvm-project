/* Measure pthread_create thread creation with different stack
   and guard sizes.

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
#include <unistd.h>
#include <support/xthread.h>

static size_t pgsize;

static void
thread_create_init (void)
{
  pgsize = sysconf (_SC_PAGESIZE);
}

static void *
thread_dummy (void *arg)
{
  return NULL;
}

static void
thread_create (int nthreads, size_t stacksize, size_t guardsize)
{
  pthread_attr_t attr;
  xpthread_attr_init (&attr);

  stacksize = stacksize * pgsize;
  guardsize = guardsize * pgsize;

  xpthread_attr_setstacksize (&attr, stacksize);
  xpthread_attr_setguardsize (&attr, guardsize);

  pthread_t ts[nthreads];

  for (int i = 0; i < nthreads; i++)
    ts[i] = xpthread_create (&attr, thread_dummy, NULL);

  for (int i = 0; i < nthreads; i++)
    xpthread_join (ts[i]);
}
