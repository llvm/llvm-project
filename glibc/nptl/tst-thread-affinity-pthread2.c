/* Separate thread test for pthread_getaffinity_np, pthread_setaffinity_np.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/* Defined for the benefit of tst-skeleton-thread-affinity.c, included
   below.  This variant runs the functions on a separate thread.  */

struct affinity_access_task
{
  pthread_t thread;
  cpu_set_t *set;
  size_t size;
  bool get;
  int result;
};

static void *
affinity_access_thread (void *closure)
{
  struct affinity_access_task *task = closure;
  if (task->get)
    task->result = pthread_getaffinity_np
      (task->thread, task->size, task->set);
  else
    task->result = pthread_setaffinity_np
      (task->thread, task->size, task->set);
  return NULL;
}

static int
run_affinity_access_thread (cpu_set_t *set, size_t size, bool get)
{
  struct affinity_access_task task =
    {
      .thread = pthread_self (),
      .set = set,
      .size = size,
      .get = get
    };
  pthread_t thr;
  int ret = pthread_create (&thr, NULL, affinity_access_thread, &task);
  if (ret != 0)
    {
      errno = ret;
      printf ("error: could not create affinity access thread: %m\n");
      abort ();
    }
  ret = pthread_join (thr, NULL);
  if (ret != 0)
    {
      errno = ret;
      printf ("error: could not join affinity access thread: %m\n");
      abort ();
    }
  if (task.result != 0)
    {
      errno = task.result;
      return -1;
    }
  return 0;
}

static int
setaffinity (size_t size, const cpu_set_t *set)
{
  return run_affinity_access_thread ((cpu_set_t *) set, size, false);
}

static int
getaffinity (size_t size, cpu_set_t *set)
{
  return run_affinity_access_thread (set, size, true);
}

#include "tst-skeleton-thread-affinity.c"
