/* Test detachability.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#define _GNU_SOURCE

#include <pthread.h>
#include <assert.h>
#include <error.h>
#include <errno.h>
#include <unistd.h>

void *
thread (void *arg)
{
  while (1)
    sched_yield ();
}

int
main (int argc, char **argv)
{
  int err;
  pthread_t tid;
  void *ret;

  err = pthread_create (&tid, 0, thread, 0);
  if (err)
    error (1, err, "pthread_create");

  err = pthread_detach (tid);
  if (err)
    error (1, err, "pthread_detach");

  err = pthread_detach (tid);
  assert (err == EINVAL);

  err = pthread_join (tid, &ret);
  assert (err == EINVAL);

  return 0;
}
